"""CERBERUS desktop launcher.

Runs the Flask app privately on this PC (127.0.0.1, random free port) and
shows it in a native window through pywebview (Microsoft Edge WebView2 on
Windows). Nothing is hosted online: detection, recordings and evidence all
stay on this machine.

    python desktop.py              # from source, uses the project folder for data
    dist\\Cerberus\\Cerberus.exe     # packaged build, uses %LOCALAPPDATA%\\Cerberus

Set CERBERUS_DATA_DIR to choose another data folder, and CERBERUS_DEBUG=1 to
enable the WebView developer tools (right-click > Inspect).
"""
import html
import os
import shutil
import sys
import threading
import time
import traceback

APP_TITLE = "CERBERUS"
APP_DIR_NAME = "Cerberus"
FROZEN = getattr(sys, "frozen", False)
# Read-only files (templates, static, model weights) live here.
BUNDLE_DIR = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
MODEL_FILES = ("best_tuned.pt", "yolo11n-pose.pt")
FINALIZE_WAIT_SEC = 120   # how long to let open recordings finish writing on exit

_mutex_handle = None      # kept alive for the life of the process


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
def _data_dir():
    """Writable folder for databases, recordings and evidence.

    From source this returns None, so the app keeps using the project folder
    exactly like `python app.py`. The packaged app cannot write next to its
    own files (Program Files is read-only), so it uses %LOCALAPPDATA%.
    """
    if os.environ.get("CERBERUS_DATA_DIR"):
        return os.environ["CERBERUS_DATA_DIR"]
    if FROZEN:
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
        return os.path.join(base, APP_DIR_NAME)
    return None


def _redirect_output(data_dir):
    """A windowed .exe has no console, so print() output would be lost and
    some libraries crash writing to a missing stderr. Send it to a log file."""
    if sys.stdout is not None and sys.stderr is not None:
        return None
    log_dir = os.path.join(data_dir or BUNDLE_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "cerberus.log")
    if os.path.exists(log_path) and os.path.getsize(log_path) > 5 * 1024 * 1024:
        os.replace(log_path, log_path + ".1")
    log = open(log_path, "a", encoding="utf-8", buffering=1)
    sys.stdout = sys.stderr = log
    print(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} CERBERUS desktop start =====")
    return log_path


def _copy_bundled_models(data_dir):
    """The app loads model weights by bare file name from the working folder.
    Copy the bundled weights there (again after an update that changed them),
    and let Ultralytics download yolo11n-pose.pt there if it was not bundled."""
    for name in MODEL_FILES:
        src = os.path.join(BUNDLE_DIR, name)
        dst = os.path.join(data_dir, name)
        if not os.path.exists(src):
            continue
        if not os.path.exists(dst) or os.path.getsize(dst) != os.path.getsize(src):
            shutil.copy2(src, dst)


def _prepare_environment():
    data_dir = _data_dir()
    if data_dir:
        os.makedirs(data_dir, exist_ok=True)
        os.environ["CERBERUS_DATA_DIR"] = data_dir
    log_path = _redirect_output(data_dir)
    if FROZEN:
        _copy_bundled_models(data_dir)
    # Relative paths in the app (model names, cerberus.db, static/violations)
    # resolve against the working folder.
    os.chdir(data_dir if FROZEN else BUNDLE_DIR)
    return log_path


# ---------------------------------------------------------------------------
# Single instance: two copies would fight over the cameras and databases.
# ---------------------------------------------------------------------------
def _already_running():
    global _mutex_handle
    if os.name != "nt":
        return False
    import ctypes
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateMutexW.restype = ctypes.c_void_p
    _mutex_handle = kernel32.CreateMutexW(None, False, "Local\\CerberusDesktopApp")
    return ctypes.get_last_error() == 183   # ERROR_ALREADY_EXISTS


def _message_box(text):
    if os.name == "nt":
        import ctypes
        ctypes.windll.user32.MessageBoxW(None, text, APP_TITLE, 0x40)
    else:
        print(text)


# ---------------------------------------------------------------------------
# Screens shown while the app starts
# ---------------------------------------------------------------------------
_PAGE = """<!doctype html><html><head><meta charset="utf-8"><style>
html,body{{margin:0;height:100%;background:#10151d;color:#edf1f4;
  font-family:'Segoe UI',system-ui,sans-serif}}
.wrap{{height:100%;display:flex;flex-direction:column;align-items:center;
  justify-content:center;gap:14px;padding:24px;box-sizing:border-box}}
h1{{margin:0;font-size:22px;letter-spacing:.18em}}
p{{margin:0;color:#9aa8b4;font-size:14px;text-align:center}}
.bar{{width:220px;height:3px;background:#202b38;border-radius:2px;overflow:hidden}}
.bar i{{display:block;width:40%;height:100%;background:#87b89a;
  animation:s 1.2s ease-in-out infinite}}
@keyframes s{{0%{{transform:translateX(-100%)}}100%{{transform:translateX(250%)}}}}
pre{{max-width:90%;max-height:55vh;overflow:auto;background:#171f2a;
  border:1px solid #303c4a;border-radius:6px;padding:12px;font-size:12px;
  color:#e07971;white-space:pre-wrap}}
</style></head><body><div class="wrap">{body}</div></body></html>"""

LOADING_HTML = _PAGE.format(body=(
    "<h1>CERBERUS</h1><div class='bar'><i></i></div>"
    "<p>Starting the detection engine&hellip;</p>"))


def _error_html(details, log_path):
    where = f"<p>Details were also written to {html.escape(log_path)}</p>" if log_path else ""
    return _PAGE.format(body=(
        "<h1>CERBERUS</h1><p>CERBERUS could not start.</p>"
        f"<pre>{html.escape(details)}</pre>{where}"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
class _Runtime:
    server = None
    app_module = None


def _boot(window, runtime, log_path):
    """Runs on a worker thread once the window is up: import the app (slow:
    it loads PyTorch), start the local server, then show the real UI."""
    try:
        from werkzeug.serving import make_server
        import app as cerberus

        server = make_server("127.0.0.1", 0, cerberus.app, threaded=True)
        runtime.server, runtime.app_module = server, cerberus
        threading.Thread(target=server.serve_forever, name="flask-server",
                         daemon=True).start()
        url = f"http://127.0.0.1:{server.server_port}/"
        print(f"[desktop] serving on {url}")
        window.load_url(url)
    except Exception:
        details = traceback.format_exc()
        print(details)
        window.load_html(_error_html(details, log_path))


def _shutdown(runtime):
    """Stop cameras, close recordings and wait for them to finish writing,
    so a normal close never leaves a half-written video behind."""
    if runtime.app_module is not None:
        try:
            runtime.app_module._shutdown()
        except Exception as e:
            print(f"[desktop] shutdown error: {e}")
    if runtime.server is not None:
        runtime.server.shutdown()
    deadline = time.time() + FINALIZE_WAIT_SEC
    for t in threading.enumerate():
        if t.name == "finalize-recording" and t.is_alive():
            t.join(timeout=max(0.0, deadline - time.time()))


def main():
    if _already_running():
        _message_box("CERBERUS is already running.")
        return
    log_path = _prepare_environment()

    import webview
    # Links with target="_blank" (e.g. Review Recordings) open in a second app
    # window that shares the login, instead of the system browser.
    webview.settings["OPEN_EXTERNAL_LINKS_IN_BROWSER"] = False

    runtime = _Runtime()
    window = webview.create_window(
        APP_TITLE, html=LOADING_HTML,
        width=1440, height=900, min_size=(1024, 680),
        background_color="#10151d",
        confirm_close=True,       # closing stops monitoring, so ask first
        text_select=True,
    )
    webview.start(_boot, (window, runtime, log_path),
                  debug=os.environ.get("CERBERUS_DEBUG") == "1")
    _shutdown(runtime)


if __name__ == "__main__":
    main()
