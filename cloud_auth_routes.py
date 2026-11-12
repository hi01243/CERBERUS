"""Cloud (Supabase) sign-in, MFA and account routes.

Used instead of auth_routes.py when cloud.enabled(). The endpoint names are
the same, so every template link and url_for() keeps working. Accounts are
identified by email; password reset uses a code sent by email; MFA is
Supabase's built-in authenticator (TOTP) support.

Login state is kept server-side: the Flask cookie only holds the Supabase
user id and email, and the Supabase client (with its tokens) lives in
memory in cloud.py.
"""
import secrets
import threading
import time

from flask import render_template, jsonify, request, session, redirect, url_for, make_response

import cloud
import cloud_sync
import state
from auth import validate_password, get_session_owner, list_owned_sessions, remove_session_owner
from auth_routes import totp_qr_data_uri
from config import DEMO_MODE, PUBLIC_ENDPOINTS, LIVE_ENDPOINTS
from session_manager import (
    _delete_session_data, _valid_session_id, _guard_static, _guard_session_access,
    _stop_live_for
)

PENDING_TTL_SEC = 15 * 60
CONFIRMATION_PHRASE = 'DELETE ACCOUNT'

# Half-finished logins (waiting for an MFA code) and MFA enrolments, keyed by
# a random token in the cookie. In memory only; they expire after 15 minutes.
_pending = {}
_pending_lock = threading.Lock()
_mfa_enrolments = {}   # user id -> {"factor_id", "uri", "secret"}


def _put_pending(**data):
    token = secrets.token_urlsafe(24)
    now = time.time()
    with _pending_lock:
        for key in [k for k, v in _pending.items() if now - v["t"] > PENDING_TTL_SEC]:
            _pending.pop(key, None)
        _pending[token] = dict(data, t=now)
    return token


def _get_pending(token):
    if not token:
        return None
    with _pending_lock:
        item = _pending.get(token)
        if item and time.time() - item["t"] > PENDING_TTL_SEC:
            _pending.pop(token, None)
            return None
        return item


def _drop_pending(token):
    with _pending_lock:
        _pending.pop(token, None)


def _verified_totp(client):
    factors = client.auth.mfa.list_factors()
    return [f for f in (factors.totp or []) if f.status == "verified"]


def _needs_mfa(client):
    aal = client.auth.mfa.get_authenticator_assurance_level()
    return aal.next_level == "aal2" and aal.current_level != "aal2"


def _verify_mfa_code(client, code):
    """True if `code` is a current code for the account's authenticator."""
    code = (code or "").strip()
    factors = _verified_totp(client)
    if not factors or not code.isdigit() or len(code) != 6:
        return False
    try:
        client.auth.mfa.challenge_and_verify({"factor_id": factors[0].id, "code": code})
        return True
    except Exception as e:
        print(f"[account] authenticator code rejected: {e}")
        return False


def _finish_login(client, user):
    session.clear()
    cloud.remember_client(user.id, client)
    session['user_id'] = user.id
    session['username'] = user.email
    try:
        session['mfa_enabled'] = bool(_verified_totp(client))
    except Exception:
        session['mfa_enabled'] = False
    print(f"[account] {user.email} signed in (cloud)")


def _no_store(response):
    response = make_response(response)
    response.headers['Cache-Control'] = 'no-store'
    return response


def init_cloud_auth_routes(app):
    """Registers the cloud authentication endpoints and login middleware."""

    @app.before_request
    def _require_login():
        if request.endpoint in PUBLIC_ENDPOINTS or request.endpoint is None:
            return
        if request.endpoint == "static":
            return _guard_static()
        if DEMO_MODE and request.endpoint == "monitor":
            return redirect(url_for("index"))
        if DEMO_MODE and request.endpoint in LIVE_ENDPOINTS:
            return jsonify({"status": "unavailable",
                            "message": "Live camera detection is disabled in the demo."}), 503
        uid = session.get("user_id")
        if uid is not None and cloud.client_for(uid) is None:
            session.clear()   # app restarted, or signed out elsewhere
            uid = None
        if not uid:
            return redirect(url_for("login"))
        return _guard_session_access()

    # ------------------------------------------------------------------
    # Sign in
    # ------------------------------------------------------------------
    @app.route('/login', methods=['GET', 'POST'])
    def login():
        if request.method == 'GET':
            return render_template('cloud_login.html')

        if request.form.get('step') == 'verify':
            token = session.get('pending_login')
            pending = _get_pending(token)
            if not pending:
                session.clear()
                return render_template('cloud_login.html',
                                       error='That sign-in expired. Please sign in again.')
            if not _verify_mfa_code(pending["client"], request.form.get('code')):
                return render_template('cloud_login.html', mfa_step=True, email=pending["email"],
                                       error='That authenticator code is invalid or expired.')
            _drop_pending(token)
            _finish_login(pending["client"], pending["user"])
            return redirect(url_for('index'))

        email = (request.form.get('email') or '').strip()
        password = request.form.get('password', '')
        client = cloud.new_client()
        try:
            response = client.auth.sign_in_with_password({"email": email, "password": password})
        except Exception as e:
            message = cloud.friendly_error(e)
            if "confirm your email" in message.lower():
                session.clear()
                session['pending_email'] = email
                return redirect(url_for('verify_email'))
            return render_template('cloud_login.html', email=email, error=message)

        try:
            needs_mfa = _needs_mfa(client)
        except Exception as e:
            return render_template('cloud_login.html', email=email, error=cloud.friendly_error(e))
        if needs_mfa:
            session.clear()
            session['pending_login'] = _put_pending(client=client, user=response.user, email=email)
            return render_template('cloud_login.html', mfa_step=True, email=email)

        _finish_login(client, response.user)
        return redirect(url_for('index'))

    # ------------------------------------------------------------------
    # Sign up (+ email confirmation code when the project requires it)
    # ------------------------------------------------------------------
    @app.route('/signup', methods=['GET', 'POST'])
    def signup():
        if request.method == 'GET':
            return render_template('cloud_signup.html')
        email = (request.form.get('email') or '').strip()
        password = request.form.get('password', '')
        enable_mfa = request.form.get('enable_mfa') == '1'
        error = validate_password(password)
        if not error and password != request.form.get('password_confirm', ''):
            error = 'The passwords do not match.'
        if error:
            return render_template('cloud_signup.html', error=error)

        client = cloud.new_client()
        try:
            response = client.auth.sign_up({"email": email, "password": password})
        except Exception as e:
            return render_template('cloud_signup.html', error=cloud.friendly_error(e))

        if response.session is not None:   # "Confirm email" is off in Supabase
            _finish_login(client, response.user)
            return redirect(url_for('mfa_setup') if enable_mfa else url_for('index'))

        session.clear()
        session['pending_email'] = email
        session['pending_enable_mfa'] = enable_mfa
        return redirect(url_for('verify_email'))

    @app.route('/verify-email', methods=['GET', 'POST'])
    def verify_email():
        email = session.get('pending_email')
        if not email:
            return redirect(url_for('login'))
        if request.method == 'GET':
            return render_template('cloud_verify_email.html', email=email)

        client = cloud.new_client()
        if request.form.get('action') == 'resend':
            try:
                client.auth.resend({"type": "signup", "email": email})
                return render_template('cloud_verify_email.html', email=email,
                                       notice='A new code is on its way.')
            except Exception as e:
                return render_template('cloud_verify_email.html', email=email,
                                       error=cloud.friendly_error(e))

        code = (request.form.get('code') or '').strip()
        try:
            response = client.auth.verify_otp({"email": email, "token": code, "type": "email"})
        except Exception as e:
            return render_template('cloud_verify_email.html', email=email,
                                   error=cloud.friendly_error(e))
        enable_mfa = session.get('pending_enable_mfa')
        _finish_login(client, response.user)
        return redirect(url_for('mfa_setup') if enable_mfa else url_for('index'))

    # ------------------------------------------------------------------
    # Forgot password: email code -> new password (-> MFA code if enabled)
    # ------------------------------------------------------------------
    @app.route('/recover-account', methods=['GET', 'POST'])
    def recover_account():
        if request.method == 'GET':
            return render_template('cloud_recover.html', step='email')
        step = request.form.get('step')

        if step == 'email':
            email = (request.form.get('email') or '').strip()
            try:
                cloud.new_client().auth.reset_password_for_email(email)
            except Exception as e:
                message = cloud.friendly_error(e)
                # Never reveal whether the address has an account; only
                # surface problems the user can act on.
                if "internet" in message or "too many" in message.lower():
                    return render_template('cloud_recover.html', step='email', error=message)
            session.clear()
            session['recover_email'] = email
            return render_template('cloud_recover.html', step='reset', email=email)

        if step == 'verify':   # MFA code after the reset code was already used
            token = session.get('pending_recovery')
            pending = _get_pending(token)
            if not pending:
                session.clear()
                return render_template('cloud_recover.html', step='email',
                                       error='That reset expired. Please start again.')
            if not _verify_mfa_code(pending["client"], request.form.get('code')):
                return render_template('cloud_login.html', mfa_step=True, email=pending["email"],
                                       error='That authenticator code is invalid or expired.')
            _drop_pending(token)
            return _set_password_and_login(pending["client"], pending["user"],
                                           pending["password"], pending["email"])

        email = session.get('recover_email')
        if not email:
            return render_template('cloud_recover.html', step='email')
        password = request.form.get('password', '')
        error = validate_password(password)
        if not error and password != request.form.get('password_confirm', ''):
            error = 'The new passwords do not match.'
        if error:
            return render_template('cloud_recover.html', step='reset', email=email, error=error)

        client = cloud.new_client()
        try:
            response = client.auth.verify_otp({
                "email": email, "token": (request.form.get('code') or '').strip(),
                "type": "recovery"})
            needs_mfa = _needs_mfa(client)
        except Exception as e:
            return render_template('cloud_recover.html', step='reset', email=email,
                                   error=cloud.friendly_error(e))

        if needs_mfa and not _verify_mfa_code(client, request.form.get('mfa_code')):
            # The email code is used up now, so ask for the MFA code on its own.
            session['pending_recovery'] = _put_pending(
                client=client, user=response.user, email=email, password=password)
            return render_template('cloud_login.html', mfa_step=True, email=email,
                                   error='This account uses MFA. Enter your authenticator code.')
        return _set_password_and_login(client, response.user, password, email)

    def _set_password_and_login(client, user, password, email):
        try:
            client.auth.update_user({"password": password})
        except Exception as e:
            return render_template('cloud_recover.html', step='email', email=email,
                                   error=cloud.friendly_error(e) + ' Request a new code and try again.')
        _finish_login(client, user)
        print(f"[account] {email} reset their password")
        return redirect(url_for('index'))

    # ------------------------------------------------------------------
    # Recovery codes do not exist for cloud accounts (email reset instead).
    # ------------------------------------------------------------------
    @app.route('/security/recovery-codes', methods=['GET', 'POST'])
    def recovery_code_settings():
        return redirect(url_for('mfa_setup'))

    # ------------------------------------------------------------------
    # Authenticator MFA
    # ------------------------------------------------------------------
    @app.route('/security/mfa', methods=['GET', 'POST'])
    def mfa_setup():
        uid = session.get('user_id')
        client = cloud.client_for(uid)
        error = None
        try:
            if _verified_totp(client):
                session['mfa_enabled'] = True
                return _no_store(render_template('mfa_setup.html', enabled=True))

            enrolment = _mfa_enrolments.get(uid)
            if request.method == 'POST' and enrolment:
                try:
                    client.auth.mfa.challenge_and_verify({
                        "factor_id": enrolment["factor_id"],
                        "code": (request.form.get('code') or '').strip()})
                    _mfa_enrolments.pop(uid, None)
                    session['mfa_enabled'] = True
                    print(f"[account] {session.get('username')} enabled MFA")
                    return redirect(url_for('index'))
                except Exception as e:
                    print(f"[account] MFA enrolment code rejected: {e}")
                    error = 'That authenticator code is invalid or expired.'

            if not enrolment:
                # Clear out half-finished factors from earlier attempts.
                for factor in (client.auth.mfa.list_factors().all or []):
                    if factor.factor_type == "totp" and factor.status != "verified":
                        client.auth.mfa.unenroll({"factor_id": factor.id})
                result = client.auth.mfa.enroll({
                    "factor_type": "totp", "issuer": "CERBERUS",
                    "friendly_name": "CERBERUS " + secrets.token_hex(3)})
                enrolment = {"factor_id": result.id, "uri": result.totp.uri,
                             "secret": result.totp.secret}
                _mfa_enrolments[uid] = enrolment
        except Exception as e:
            return _no_store(render_template('mfa_setup.html', enabled=False, secret=None,
                                             qr_data_uri=None, error=cloud.friendly_error(e)))

        return _no_store(render_template(
            'mfa_setup.html', enabled=False, secret=enrolment["secret"],
            qr_data_uri=totp_qr_data_uri(enrolment["uri"]), error=error))

    @app.route('/security/mfa/status')
    def mfa_status():
        return jsonify({'enabled': bool(session.get('mfa_enabled'))})

    @app.route('/security/mfa/disable', methods=['POST'])
    def disable_mfa():
        client = cloud.client_for(session.get('user_id'))
        code = (request.get_json(silent=True) or {}).get('code')
        try:
            factors = _verified_totp(client)
            if not factors or not _verify_mfa_code(client, code):
                return jsonify({'message': 'That authenticator code is invalid or expired.'}), 403
            for factor in factors:
                client.auth.mfa.unenroll({"factor_id": factor.id})
        except Exception as e:
            return jsonify({'message': cloud.friendly_error(e)}), 502
        session['mfa_enabled'] = False
        print(f"[account] {session.get('username')} disabled MFA")
        return jsonify({'status': 'disabled'})

    # ------------------------------------------------------------------
    # Delete account: cloud images + account (rows cascade), then local data
    # ------------------------------------------------------------------
    @app.route('/account/delete', methods=['GET', 'POST'])
    def delete_account():
        uid = session.get('user_id')
        email = session.get('username')
        client = cloud.client_for(uid)

        csrf_token = session.get('account_delete_csrf_token')
        if not csrf_token:
            csrf_token = secrets.token_urlsafe(32)
            session['account_delete_csrf_token'] = csrf_token
        current_owned = get_session_owner(state.CURRENT_SESSION) == uid
        active_owned_session = bool(state.monitoring_active and current_owned)
        mfa_enabled = bool(session.get('mfa_enabled'))

        def render_page(error=None, status=200):
            return render_template(
                'delete_account.html', username=email, csrf_token=csrf_token,
                active_owned_session=active_owned_session, mfa_enabled=mfa_enabled,
                confirmation_phrase=CONFIRMATION_PHRASE, error=error), status

        if request.method == 'GET':
            return render_page()
        if not secrets.compare_digest(csrf_token, request.form.get('csrf_token', '')):
            return render_page('This request expired. Reload the page and try again.', 400)
        if request.form.get('confirm_username', '').strip().lower() != (email or '').lower():
            return render_page('The email confirmation is incorrect.', 400)
        if request.form.get('confirm_phrase') != CONFIRMATION_PHRASE:
            return render_page(f'Type "{CONFIRMATION_PHRASE}" exactly to confirm deletion.', 400)
        if active_owned_session:
            return render_page('Stop monitoring before deleting this account.', 400)

        checker = cloud.new_client()   # re-check the password without touching this login
        try:
            checker.auth.sign_in_with_password({"email": email,
                                                "password": request.form.get('password', '')})
            checker.auth.sign_out({"scope": "local"})
        except Exception as e:
            return render_page(cloud.friendly_error(e), 400)
        if mfa_enabled and not _verify_mfa_code(client, request.form.get('totp_code')):
            return render_page('The authenticator code is invalid or expired.', 400)

        try:
            cloud_sync.delete_all_cloud_data(client, uid)
            client.rpc("delete_my_account").execute()
        except Exception as e:
            print(f"[account] could not delete cloud account {email}: {e}")
            return render_page('The cloud account could not be deleted, so nothing was removed. '
                               + cloud.friendly_error(e), 502)

        owned = set(list_owned_sessions(uid))
        for session_id in owned:
            if _valid_session_id(session_id):
                _delete_session_data(session_id, cloud_cleanup=False)
            remove_session_owner(session_id)
        if current_owned:
            remove_session_owner(state.CURRENT_SESSION)
        cloud.forget_client(uid, sign_out=False)
        _mfa_enrolments.pop(uid, None)
        print(f"[account] {email} deleted their cloud account and {len(owned)} local session(s)")
        session.clear()
        return redirect(url_for('login'))

    @app.route('/logout', methods=['POST'])
    def logout():
        uid = session.get('user_id')
        _stop_live_for(uid)   # monitoring and cameras stop on sign-out
        if uid:
            cloud.forget_client(uid)
            _mfa_enrolments.pop(uid, None)
        session.clear()
        return redirect(url_for('login'))
