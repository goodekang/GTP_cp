Param(
  [string]$Root = "data/brca",
  [string]$OutDir = "data/brca/raw/gdc_wsi",
  [string]$GdcClient = "gdc-client.exe",
  [string]$TokenFile = ""
)

$ErrorActionPreference = "Stop"

$Manifest = Join-Path $Root "manifests\gdc_manifest.tsv"

if (!(Test-Path $Manifest)) {
  Write-Host "[error] manifest not found: $Manifest"
  Write-Host "Run: gtp-brca-manifest --root $Root"
  Write-Host "Then: gtp-brca-gdc-manifest --root $Root --svs-only"
  exit 1
}

if (!(Test-Path $GdcClient)) {
  Write-Host "[error] gdc-client not found: $GdcClient"
  Write-Host "Download gdc-client for Windows from GDC, place it next to this script, or pass -GdcClient."
  exit 1
}

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

Write-Host "[info] manifest: $Manifest"
Write-Host "[info] outdir:   $OutDir"

$argsList = @("download", "-m", $Manifest, "-d", $OutDir)
if ($TokenFile -ne "") {
  if (!(Test-Path $TokenFile)) { throw "Token file not found: $TokenFile" }
  $argsList += @("-t", $TokenFile)
  Write-Host "[info] token:    $TokenFile"
} else {
  Write-Host "[info] token:    (none)"
}

Write-Host "[run] $GdcClient $($argsList -join ' ')"
& $GdcClient @argsList


