Param(
  [string]$Version = "1.6.1",
  [string]$OutPath = "gdc-client.exe"
)

$ErrorActionPreference = "Stop"

# Official release assets are hosted by NCI GDC.
# If the exact URL changes, use the GDC website to locate the latest Windows binary.
$Url = "https://gdc.cancer.gov/files/public/file/gdc-client_v$Version_Windows_x64.zip"

$Tmp = Join-Path $env:TEMP "gdc-client-$Version.zip"
$TmpDir = Join-Path $env:TEMP "gdc-client-$Version"

Write-Host "[download] $Url"
Invoke-WebRequest -Uri $Url -OutFile $Tmp

if (Test-Path $TmpDir) { Remove-Item -Recurse -Force $TmpDir }
New-Item -ItemType Directory -Force -Path $TmpDir | Out-Null

Expand-Archive -Path $Tmp -DestinationPath $TmpDir -Force

$exe = Get-ChildItem -Path $TmpDir -Recurse -Filter "gdc-client.exe" | Select-Object -First 1
if ($null -eq $exe) { throw "gdc-client.exe not found inside zip; URL/version may have changed." }

Copy-Item $exe.FullName $OutPath -Force
Write-Host "[ok] wrote: $OutPath"


