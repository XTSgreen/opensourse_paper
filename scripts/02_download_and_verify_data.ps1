param([switch]$DownloadRaw)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
if ($DownloadRaw) {
    python data/download/download_public_data.py
} else {
    python data/download/download_public_data.py --dry-run
    Write-Host "Frozen processed panels are packaged locally. Pass -DownloadRaw to fetch the optional full GEO archives."
}
python scripts/reproduce.py prepare
