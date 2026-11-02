; Inno Setup 6 Script for Biceps Brachii Ultrasound & Biomechanics Studio
; Generates a professional single-file Windows Installer Wizard (.exe)

#ifndef AppVersion
#define AppVersion "2.1.0"
#endif

[Setup]
AppId={{E68F8A8B-2F67-4DF8-8B7C-A89E62C4E3D1}
AppName=Biceps Brachii Pipeline Studio
AppVersion={#AppVersion}
AppVerName=Biceps Brachii Pipeline Studio v{#AppVersion}
AppPublisher=National Taiwan University / Asia University / Probefix
AppPublisherURL=https://github.com/Corneliox/Segmentation_Configuration_Bicheps_Brachii
AppSupportURL=https://github.com/Corneliox/Segmentation_Configuration_Bicheps_Brachii/issues
AppUpdatesURL=https://github.com/Corneliox/Segmentation_Configuration_Bicheps_Brachii/releases
DefaultDirName={autopf}\BicepsBrachiiStudio
DefaultGroupName=Biceps Brachii Pipeline Studio
DisableProgramGroupPage=yes
OutputDir=dist_installer
OutputBaseFilename=Biceps_Brachii_Studio_Setup_{#AppVersion}
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
UninstallDisplayIcon={app}\Biceps_Brachii_Pipeline_Studio.exe
CloseApplications=yes
RestartApplications=no

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
; Main application binaries and dependencies
Source: "dist\Biceps_Brachii_Pipeline_Studio\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\Biceps Brachii Pipeline Studio"; Filename: "{app}\Biceps_Brachii_Pipeline_Studio.exe"
Name: "{group}\Uninstall Biceps Brachii Studio"; Filename: "{uninstallexe}"
Name: "{autodesktop}\Biceps Brachii Pipeline Studio"; Filename: "{app}\Biceps_Brachii_Pipeline_Studio.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\Biceps_Brachii_Pipeline_Studio.exe"; Description: "{cm:LaunchProgram,Biceps Brachii Pipeline Studio}"; Flags: nowait postinstall skipifsilent
