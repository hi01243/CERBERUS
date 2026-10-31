"""Authentication, MFA, and account security routes for Cerberus."""
import base64
from io import BytesIO
import secrets
from urllib.parse import urlencode
from flask import (
    render_template, jsonify, request, session,
    redirect, url_for, make_response
)
try:
    import qrcode
    from qrcode.image.svg import SvgPathImage
except ImportError:
    qrcode = None
    SvgPathImage = None

from config import DEMO_MODE, PUBLIC_ENDPOINTS, LIVE_ENDPOINTS
from auth import (
    create_user, get_user_by_id, verify_user,
    generate_totp_secret, get_totp_secret, save_totp_secret, verify_totp,
    delete_user, generate_recovery_codes, verify_recovery_code,
    reset_password_with_recovery_code, validate_password,
    user_count, first_user_id_excluding, get_session_owner, list_owned_sessions
)
import state
from session_manager import (
    _assign_unowned_sessions, _delete_session_data, _valid_session_id,
    _guard_static, _guard_session_access
)


def init_auth_routes(app):
    """Registers authentication endpoints and security middleware on Flask app."""

    @app.before_request
    def _require_login():
        if request.endpoint in PUBLIC_ENDPOINTS or request.endpoint is None:
            return
        if request.endpoint == "static":
            return _guard_static()
        if DEMO_MODE and request.endpoint == "monitor":
            return redirect(url_for("index"))
        if DEMO_MODE and request.endpoint in LIVE_ENDPOINTS:
            return jsonify({
                "status": "unavailable",
                "message": "Live camera detection is disabled in the Render free demo.",
            }), 503
        if session.get("user_id") is not None and not get_user_by_id(session["user_id"]):
            session.clear()
        if not session.get("user_id"):
            return redirect(url_for("login"))
        return _guard_session_access()

    @app.route('/login', methods=['GET', 'POST'])
    def login():
        error = None
        if request.method == 'POST':
            pending_user_id = session.get('pending_login_user_id')
            if pending_user_id and request.form.get('step') == 'verify':
                user = get_user_by_id(pending_user_id)
                code = request.form.get('code', '')
                if user and (verify_totp(user.get('totp_secret'), code)
                             or verify_recovery_code(user['id'], code)):
                    session.clear()
                    session['user_id'] = user['id']
                    session['username'] = user['username']
                    return redirect(url_for('index'))
                error = 'Authenticator or recovery code is invalid, expired, or already used.'
                return render_template('login.html', error=error,
                                       verification_username=session.get('pending_login_username'))

            username = (request.form.get('username') or '').strip()
            user = verify_user(username, request.form.get('password', ''))
            if user:
                if user.get('totp_secret'):
                    session.clear()
                    session['pending_login_user_id'] = user['id']
                    session['pending_login_username'] = user['username']
                    return render_template('login.html', verification_username=user['username'])
                session.clear()
                session['user_id'] = user['id']
                session['username'] = user['username']
                return redirect(url_for('index'))
            else:
                error = 'Incorrect username or password.'
        return render_template('login.html', error=error)

    @app.route('/signup', methods=['GET', 'POST'])
    def signup():
        error = None
        if request.method == 'POST':
            username = (request.form.get('username') or '').strip()
            password = request.form.get('password', '')
            confirmation = request.form.get('password_confirm', '')
            if password != confirmation:
                error = 'The passwords do not match.'
                user_id = None
            else:
                user_id, error = create_user(username, password)
            if user_id:
                if user_count() == 1:
                    _assign_unowned_sessions(user_id)
                session.clear()
                session['user_id'] = user_id
                session['username'] = username
                recovery_codes = generate_recovery_codes(user_id)
                next_url = url_for('mfa_setup') if request.form.get('enable_mfa') == '1' else url_for('index')
                response = make_response(render_template(
                    'recovery_codes.html', recovery_codes=recovery_codes,
                    next_url=next_url, error=None,
                ))
                response.headers['Cache-Control'] = 'no-store'
                return response
        return render_template('signup.html', error=error)

    @app.route('/recover-account', methods=['GET', 'POST'])
    def recover_account():
        error = None
        if request.method == 'POST':
            username = (request.form.get('username') or '').strip()
            recovery_code = request.form.get('recovery_code', '')
            password = request.form.get('password', '')
            confirmation = request.form.get('password_confirm', '')
            password_error = validate_password(password)
            if password_error:
                error = password_error
            elif password != confirmation:
                error = 'The new passwords do not match.'
            else:
                user = reset_password_with_recovery_code(username, recovery_code, password)
                if user:
                    session.clear()
                    session['user_id'] = user['id']
                    session['username'] = user['username']
                    return redirect(url_for('index'))
                error = 'The username or recovery code is invalid or the code has already been used.'
        return render_template('recover_account.html', error=error)

    @app.route('/security/recovery-codes', methods=['GET', 'POST'])
    def recovery_code_settings():
        user = get_user_by_id(session.get('user_id'))
        if not user:
            session.clear()
            return redirect(url_for('login'))
        error = None
        recovery_codes = None
        if request.method == 'POST':
            if not verify_user(user['username'], request.form.get('password', '')):
                error = 'The current password is incorrect.'
            else:
                recovery_codes = generate_recovery_codes(user['id'])
        response = make_response(render_template(
            'recovery_codes.html', recovery_codes=recovery_codes,
            next_url=url_for('index'), error=error,
        ))
        response.headers['Cache-Control'] = 'no-store'
        return response

    @app.route('/security/mfa', methods=['GET', 'POST'])
    def mfa_setup():
        user_id = session.get('user_id')
        active_secret = get_totp_secret(user_id)
        error = None
        pending_secret = session.get('pending_totp_secret')

        if request.method == 'POST' and not active_secret:
            if pending_secret and verify_totp(pending_secret, request.form.get('code')):
                save_totp_secret(user_id, pending_secret)
                session.pop('pending_totp_secret', None)
                return redirect(url_for('index'))
            error = 'That authenticator code is invalid or expired.'

        if not active_secret and not pending_secret:
            pending_secret = generate_totp_secret()
            session['pending_totp_secret'] = pending_secret

        qr_data_uri = None
        if pending_secret and not active_secret and qrcode is not None and SvgPathImage is not None:
            provisioning_uri = 'otpauth://totp/CERBERUS?' + urlencode({
                'secret': pending_secret,
                'issuer': 'CERBERUS',
                'algorithm': 'SHA1',
                'digits': 6,
                'period': 30,
            })
            qr = qrcode.QRCode(border=4, box_size=8)
            qr.add_data(provisioning_uri)
            qr.make(fit=True)
            image = qr.make_image(image_factory=SvgPathImage)
            buffer = BytesIO()
            image.save(buffer)
            qr_data_uri = 'data:image/svg+xml;base64,' + base64.b64encode(
                buffer.getvalue()
            ).decode('ascii')

        response = make_response(render_template(
            'mfa_setup.html', enabled=bool(active_secret), secret=pending_secret,
            qr_data_uri=qr_data_uri, error=error,
        ))
        response.headers['Cache-Control'] = 'no-store'
        return response

    @app.route('/security/mfa/status')
    def mfa_status():
        return jsonify({'enabled': bool(get_totp_secret(session.get('user_id')))})

    @app.route('/security/mfa/disable', methods=['POST'])
    def disable_mfa():
        user_id = session.get('user_id')
        secret = get_totp_secret(user_id)
        data = request.get_json(silent=True) or {}
        code = data.get('code')
        if not secret or (not verify_totp(secret, code)
                          and not verify_recovery_code(user_id, code)):
            return jsonify({'message': 'Authenticator or recovery code is invalid, expired, or already used.'}), 403
        save_totp_secret(user_id, None)
        return jsonify({'status': 'disabled'})

    @app.route('/account/delete', methods=['GET', 'POST'])
    def delete_account():
        confirmation_phrase = 'DELETE ACCOUNT'
        user_id = session.get('user_id')
        user = get_user_by_id(user_id)
        if not user:
            session.clear()
            return redirect(url_for('login'))

        csrf_token = session.get('account_delete_csrf_token')
        if not csrf_token:
            csrf_token = secrets.token_urlsafe(32)
            session['account_delete_csrf_token'] = csrf_token

        current_owned = get_session_owner(state.CURRENT_SESSION) == user_id
        active_owned_session = bool(state.monitoring_active and current_owned)
        mfa_secret = get_totp_secret(user_id)

        def render_page(error=None, status=200):
            return render_template(
                'delete_account.html', username=user['username'], csrf_token=csrf_token,
                active_owned_session=active_owned_session, mfa_enabled=bool(mfa_secret),
                confirmation_phrase=confirmation_phrase, error=error,
            ), status

        if request.method == 'GET':
            return render_page()

        submitted_token = request.form.get('csrf_token', '')
        if not secrets.compare_digest(csrf_token, submitted_token):
            return render_page('This request expired. Reload the page and try again.', 400)
        if request.form.get('confirm_username') != user['username'] or not verify_user(
                user['username'], request.form.get('password', '')):
            return render_page('The username confirmation or password is incorrect.', 400)
        if request.form.get('confirm_phrase') != confirmation_phrase:
            return render_page(f'Type "{confirmation_phrase}" exactly to confirm deletion.', 400)
        if mfa_secret and not verify_totp(mfa_secret, request.form.get('totp_code')):
            return render_page('The authenticator code is invalid or expired.', 400)
        if active_owned_session:
            return render_page('Stop monitoring before deleting this account.', 400)

        owned_sessions = set(list_owned_sessions(user_id))
        if current_owned:
            owned_sessions.add(state.CURRENT_SESSION)

        try:
            for session_id in owned_sessions:
                if _valid_session_id(session_id):
                    _delete_session_data(session_id)

            replacement_owner = first_user_id_excluding(user_id)
            if not delete_user(
                    user_id,
                    replacement_session_id=state.CURRENT_SESSION if current_owned else None,
                    replacement_owner_id=replacement_owner):
                raise RuntimeError('Account record was not deleted.')
        except Exception as error:
            print(f"[account] could not delete account #{user_id}: {error}")
            return render_page('Account deletion could not be completed. Some session data may already have been removed.', 500)

        print(f"[account] {user['username']} deleted their account and {len(owned_sessions)} owned session(s)")
        session.clear()
        return redirect(url_for('login'))

    @app.route('/logout', methods=['POST'])
    def logout():
        session.clear()
        return redirect(url_for('login'))

