; Kor Studio installer (Inno Setup 6). Built by scripts\package.ps1, which
; passes the version and the packed release folder:
;   iscc /DAppVersion=0.7.1 /DSourceDir=..\build\release\kor-studio-0.7.1-windows-x64 installer\kor-studio.iss
; Installs for the current user without administrator rights; the setup
; offers an all-users install too.

#ifndef AppVersion
  #define AppVersion "0.7.1"
#endif
#ifndef SourceDir
  #define SourceDir "..\build\release\kor-studio-" + AppVersion + "-windows-x64"
#endif
#ifndef OutputDir
  #define OutputDir "..\build\release"
#endif

[Setup]
AppId={{B76E7BB1-6ABC-45FD-B3D9-2C6667353CF5}
AppName=Kor Studio
AppVersion={#AppVersion}
AppVerName=Kor Studio {#AppVersion}
AppPublisher=Hydra-Ai-tr
AppPublisherURL=https://github.com/Hydra-Ai-tr/kor-studio
AppSupportURL=https://github.com/Hydra-Ai-tr/kor-studio/issues
AppUpdatesURL=https://github.com/Hydra-Ai-tr/kor-studio/releases
DefaultDirName={autopf}\Kor Studio
DefaultGroupName=Kor Studio
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
LicenseFile={#SourceDir}\LICENSE
SetupIconFile=..\assets\kor.ico
UninstallDisplayIcon={app}\kor-studio.exe
OutputDir={#OutputDir}
OutputBaseFilename=kor-studio-{#AppVersion}-setup
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ChangesEnvironment=yes
CloseApplications=yes

[Languages]
Name: "en"; MessagesFile: "compiler:Default.isl"
Name: "tr"; MessagesFile: "compiler:Languages\Turkish.isl"
Name: "ru"; MessagesFile: "compiler:Languages\Russian.isl"

[CustomMessages]
en.AddToPath=Add kor-cli to the PATH (use it from any terminal)
tr.AddToPath=kor-cli'yi PATH'e ekle (her terminalden kullan)
ru.AddToPath=Добавить kor-cli в PATH (запуск из любого терминала)

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "addtopath"; Description: "{cm:AddToPath}"; Flags: unchecked

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

[Icons]
Name: "{autoprograms}\Kor Studio"; Filename: "{app}\kor-studio.exe"
Name: "{autodesktop}\Kor Studio"; Filename: "{app}\kor-studio.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\kor-studio.exe"; Description: "{cm:LaunchProgram,Kor Studio}"; Flags: nowait postinstall skipifsilent

[Code]
const
  UserEnv = 'Environment';
  SystemEnv = 'SYSTEM\CurrentControlSet\Control\Session Manager\Environment';

function EnvRoot: Integer;
begin
  if IsAdminInstallMode then Result := HKEY_LOCAL_MACHINE else Result := HKEY_CURRENT_USER;
end;

function EnvKey: String;
begin
  if IsAdminInstallMode then Result := SystemEnv else Result := UserEnv;
end;

function PathHas(const Paths, Dir: String): Boolean;
begin
  Result := Pos(';' + Uppercase(Dir) + ';', ';' + Uppercase(Paths) + ';') > 0;
end;

procedure AddToPath(const Dir: String);
var
  Paths: String;
begin
  if not RegQueryStringValue(EnvRoot, EnvKey, 'Path', Paths) then Paths := '';
  if PathHas(Paths, Dir) then Exit;
  if (Paths <> '') and (Copy(Paths, Length(Paths), 1) <> ';') then Paths := Paths + ';';
  RegWriteExpandStringValue(EnvRoot, EnvKey, 'Path', Paths + Dir);
end;

procedure RemoveFromPath(const Dir: String);
var
  Paths: String;
  P: Integer;
begin
  if not RegQueryStringValue(EnvRoot, EnvKey, 'Path', Paths) then Exit;
  Paths := ';' + Paths + ';';
  P := Pos(';' + Uppercase(Dir) + ';', Uppercase(Paths));
  if P = 0 then Exit;
  Delete(Paths, P, Length(Dir) + 1);
  Paths := Copy(Paths, 2, Length(Paths) - 2);
  RegWriteExpandStringValue(EnvRoot, EnvKey, 'Path', Paths);
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if (CurStep = ssPostInstall) and WizardIsTaskSelected('addtopath') then
    AddToPath(ExpandConstant('{app}'));
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usPostUninstall then
    RemoveFromPath(ExpandConstant('{app}'));
end;
