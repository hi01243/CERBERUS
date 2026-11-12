# PyInstaller spec for the CERBERUS desktop app.
# Build with build_desktop.bat (or: pyinstaller cerberus.spec --noconfirm).
# Output: dist\Cerberus\Cerberus.exe  (one folder; ship the whole folder)
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

ROOT = os.path.abspath(SPECPATH)

datas = [(os.path.join(ROOT, "templates"), "templates")]

# Static files, but never the evidence images under static/violations:
# those are student data and must not end up inside the installer.
for folder, dirs, files in os.walk(os.path.join(ROOT, "static")):
    dirs[:] = [d for d in dirs if d != "violations"]
    rel = os.path.relpath(folder, ROOT)
    for f in files:
        datas.append((os.path.join(folder, f), rel))

# Model weights. yolo11n-pose.pt is optional: without it the app downloads
# it on first camera start, which needs internet once.
for model in ("best_tuned.pt", "yolo11n-pose.pt"):
    if os.path.exists(os.path.join(ROOT, model)):
        datas.append((os.path.join(ROOT, model), "."))
    elif model == "best_tuned.pt":
        raise SystemExit("best_tuned.pt is missing: put the phone model next to app.py")

# Supabase project settings (URL + PUBLIC key). Without this file the build
# uses local accounts. cloud.py refuses a secret/service_role key.
if os.path.exists(os.path.join(ROOT, "cloud_config.json")):
    datas.append((os.path.join(ROOT, "cloud_config.json"), "."))
    print("[spec] bundling cloud_config.json -> cloud accounts")
else:
    print("[spec] no cloud_config.json -> local accounts")

# Ultralytics reads its own YAML files at runtime (bytetrack.yaml, model cfgs).
datas += collect_data_files("ultralytics")
datas += collect_data_files("webview")

hiddenimports = (
    collect_submodules("webview")
    + collect_submodules("ultralytics")
    + collect_submodules("supabase")
    + collect_submodules("supabase_auth")
    + collect_submodules("postgrest")
    + collect_submodules("storage3")
    + ["lap", "qrcode", "PIL"]
)

a = Analysis(
    [os.path.join(ROOT, "desktop.py")],
    pathex=[ROOT],
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["tkinter", "matplotlib.tests", "IPython", "jupyter", "notebook"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Cerberus",
    console=False,          # no black console window; output goes to logs\cerberus.log
    upx=False,              # UPX breaks some torch/CUDA DLLs
    icon=os.path.join(ROOT, "static", "cerberus.ico")
         if os.path.exists(os.path.join(ROOT, "static", "cerberus.ico")) else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="Cerberus",
)
