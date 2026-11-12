; =====================================================================
;  CERBERUS Windows installer (Inno Setup 6).
;  Build:  build_desktop.bat   (runs PyInstaller, then this script)
;     or:  ISCC.exe installer.iss   after dist\Cerberus exists.
;  Output: installer\CerberusSetup.exe
; =====================================================================

#define AppName      "CERBERUS"
#define AppVersion   "1.0.0"
#define AppExe       "Cerberus.exe"
#define AppDataDir   "Cerberus"

[Setup]
; Keep this AppId the same in every release, so new versions upgrade in place.
AppId={{6E3B0C51-7F2A-4C8E-9B1D-2A4F5C7E9D10}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppName}
VersionInfoVersion={#AppVersion}
VersionInfoDescription={#AppName} exam proctoring setup
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
; Installs for the current user without admin rights (school PCs); the
; first page lets an administrator install for all users instead.
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
SetupIconFile=static\cerberus.ico
UninstallDisplayIcon={app}\{#AppExe}
UninstallDisplayName={#AppName}
WizardStyle=modern
OutputDir=installer
OutputBaseFilename=CerberusSetup
Compression=lzma2/max
SolidCompression=yes
LZMAUseSeparateProcess=yes
LZMANumBlockThreads=4
; The GPU build is several GB. If the compressed setup is too big for one
; file, set DiskSpanning=yes: you then get CerberusSetup.exe plus .bin
; files that must be kept together in the same folder.
DiskSpanning=no
; Ask to close CERBERUS if it is running (desktop.py creates this mutex).
AppMutex=CerberusDesktopApp
CloseApplications=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "dist\Cerberus\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent

[Code]
const
  WebView2Key = 'Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}';

{ CERBERUS draws its window with Microsoft Edge WebView2 (built into
  Windows 11, and on most Windows 10 PCs). Warn if it is missing. }
function HasWebView2(): Boolean;
var
  Version: String;
begin
  Result :=
    (RegQueryStringValue(HKLM, 'SOFTWARE\WOW6432Node\' + WebView2Key, 'pv', Version) and (Version <> '') and (Version <> '0.0.0.0')) or
    (RegQueryStringValue(HKLM, 'SOFTWARE\' + WebView2Key, 'pv', Version) and (Version <> '') and (Version <> '0.0.0.0')) or
    (RegQueryStringValue(HKCU, 'Software\' + WebView2Key, 'pv', Version) and (Version <> '') and (Version <> '0.0.0.0'));
end;

function InitializeSetup(): Boolean;
begin
  Result := True;
  if not HasWebView2() then
    if MsgBox('CERBERUS needs the Microsoft Edge WebView2 Runtime, which was not found on this PC.' + #13#10 + #13#10 +
              'You can install it for free from:' + #13#10 +
              'https://developer.microsoft.com/microsoft-edge/webview2/' + #13#10 + #13#10 +
              'Continue installing CERBERUS anyway?', mbConfirmation, MB_YESNO) = IDNO then
      Result := False;
end;

{ Recordings, flag images and the local databases live in
  %LOCALAPPDATA%\Cerberus, not in the install folder. Keep them unless the
  user explicitly asks to remove them. }
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  DataDir: String;
begin
  if CurUninstallStep = usPostUninstall then
  begin
    DataDir := ExpandConstant('{localappdata}\{#AppDataDir}');
    if DirExists(DataDir) and not UninstallSilent() then
      if MsgBox('Also delete the recordings, flag images and session data stored on this PC?' + #13#10 + #13#10 +
                DataDir + #13#10 + #13#10 +
                'Choose No to keep them (for example to reinstall later). ' +
                'Flags already uploaded to your cloud account are not affected.',
                mbConfirmation, MB_YESNO or MB_DEFBUTTON2) = IDYES then
        DelTree(DataDir, True, True, True);
  end;
end;
