# Packs the Windows release from a finished build into build/release:
#   kor-studio-<version>-windows-x64/      the portable folder
#   kor-studio-<version>-windows-x64.zip   the same folder zipped
#   kor-studio-<version>-setup.exe         the installer (needs Inno Setup 6)
#
# Signing (optional): set KOR_SIGN_PFX to a .pfx code-signing certificate and
# KOR_SIGN_PASSWORD to its password. signtool from the Windows SDK then signs
# both programs and the installer, with a timestamp so the signature outlives
# the certificate.
#
#   cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
#   cmake --build build
#   powershell -ExecutionPolicy Bypass -File scripts\package.ps1
param(
    [string]$BuildDir = "build",
    [string]$Version = "",
    [switch]$NoInstaller
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not $Version) {
    $Version = (Select-String -Path CMakeLists.txt -Pattern 'project\(KorStudio VERSION ([0-9.]+)').Matches[0].Groups[1].Value
}
$bin = Join-Path $BuildDir "bin"
foreach ($exe in "kor-studio.exe", "kor-cli.exe") {
    if (-not (Test-Path (Join-Path $bin $exe))) { throw "$exe is missing in ${bin}: build first (cmake --build $BuildDir)" }
}

function Find-SignTool {
    $cmd = Get-Command signtool.exe -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $kits = "${env:ProgramFiles(x86)}\Windows Kits\10\bin"
    if (Test-Path $kits) {
        $found = Get-ChildItem $kits -Directory | Sort-Object Name -Descending |
            ForEach-Object { Join-Path $_.FullName "x64\signtool.exe" } | Where-Object { Test-Path $_ } | Select-Object -First 1
        if ($found) { return $found }
    }
    return $null
}

$sign = $null
if ($env:KOR_SIGN_PFX) {
    if (-not (Test-Path $env:KOR_SIGN_PFX)) { throw "KOR_SIGN_PFX points to a missing file: $env:KOR_SIGN_PFX" }
    $sign = Find-SignTool
    if (-not $sign) { throw "KOR_SIGN_PFX is set but signtool.exe was not found (install the Windows SDK)" }
}
function Sign-File([string]$file) {
    if (-not $sign) { return }
    & $sign sign /f $env:KOR_SIGN_PFX /p $env:KOR_SIGN_PASSWORD /fd sha256 /tr http://timestamp.digicert.com /td sha256 /d "Kor Studio" $file
    if ($LASTEXITCODE -ne 0) { throw "signing $file failed" }
}

$out = Join-Path $root "build\release"
$name = "kor-studio-$Version-windows-x64"
$dir = Join-Path $out $name
if (Test-Path $dir) { Remove-Item $dir -Recurse -Force }
foreach ($d in $dir, "$dir\assets", "$dir\community", "$dir\examples", "$dir\plugins", "$dir\python\kor") {
    New-Item -ItemType Directory -Force $d | Out-Null
}
Copy-Item "$bin\kor-studio.exe", "$bin\kor-cli.exe", LICENSE, README.md $dir
Copy-Item assets\hero.png "$dir\assets"
Copy-Item community\index.json "$dir\community"
Copy-Item plugins\*.py "$dir\plugins"
Copy-Item python\kor\*.py "$dir\python\kor"
Get-ChildItem examples -File | Where-Object { $_.Extension -in ".txt", ".csv", ".py" } | Copy-Item -Destination "$dir\examples"
Sign-File "$dir\kor-studio.exe"
Sign-File "$dir\kor-cli.exe"

$zip = "$dir.zip"
if (Test-Path $zip) { Remove-Item $zip -Force }
Compress-Archive -Path $dir -DestinationPath $zip
Write-Host "wrote $zip"

if ($NoInstaller) { return }
$iscc = (Get-Command iscc.exe -ErrorAction SilentlyContinue).Source
foreach ($p in "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe", "$env:ProgramFiles\Inno Setup 6\ISCC.exe", "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe") {
    if (-not $iscc -and (Test-Path $p)) { $iscc = $p }
}
if (-not $iscc) {
    Write-Host "Inno Setup 6 not found: skipped the installer (https://jrsoftware.org/isinfo.php)"
    return
}
& $iscc /Qp "/DAppVersion=$Version" "/DSourceDir=$dir" "/DOutputDir=$out" installer\kor-studio.iss
if ($LASTEXITCODE -ne 0) { throw "Inno Setup failed" }
$setup = Join-Path $out "kor-studio-$Version-setup.exe"
Sign-File $setup
Write-Host "wrote $setup"
