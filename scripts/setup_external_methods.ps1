param([switch]$CreateEnvironments)
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

<#
This script documents the reproducible external-method setup. It requires Git and
Conda. Each source checkout is pinned by external_methods/SOURCE_PINS.yaml.
The current local run used equivalent environments recorded under
external_methods/environments/; those runtime directories are excluded from Git.
#>

$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$sourceRoot = Join-Path $root "external_methods\sources"
New-Item -ItemType Directory -Force -Path $sourceRoot | Out-Null

$repositories = @{
    "wot" = "https://github.com/broadinstitute/wot.git"
    "moscot" = "https://github.com/theislab/moscot.git"
    "lineageot" = "https://github.com/aforr/LineageOT.git"
    "tigon" = "https://github.com/yutongo/TIGON.git"
    "mioflow" = "https://github.com/KrishnaswamyLab/MIOFlow.git"
    "cellrank2" = "https://github.com/scverse/cellrank.git"
    "prescient" = "https://github.com/gifford-lab/prescient.git"
}
$commits = @{
    "wot" = "ca5e94f05699997b01cf5ae13383f9810f0613f6"
    "moscot" = "440093ccbb8e70de209157d91da839c55b897821"
    "lineageot" = "6081b402074f7e5934e729e81669aef430219da8"
    "tigon" = "1ed92cfcc250415fc01b4d344a308b0680cc9635"
    "mioflow" = "36365403d0f23cc3ad1065781c7331bf81debf4e"
    "cellrank2" = "d7191d75684c86b58adbb317c8aae7d06f2682f3"
    "prescient" = "50971c7d495e8763eaa60f83af91f51555ed7ece"
}

foreach ($name in $repositories.Keys) {
    $destination = Join-Path $sourceRoot $name
    if (-not (Test-Path (Join-Path $destination ".git"))) {
        git clone $repositories[$name] $destination
    }
    git -C $destination fetch --quiet --all --tags
    git -C $destination checkout --quiet --detach $commits[$name]
    $head = (git -C $destination rev-parse HEAD).Trim()
    if ($head -ne $commits[$name]) {
        throw "Pinned commit verification failed for ${name}: expected $($commits[$name]), got $head"
    }
}

Write-Host "Source repositories are checked out at the exact commits recorded in SOURCE_PINS.yaml. Create the four method environments from external_methods/environments/*.yaml."
Write-Host "The benchmark runner validates method status and retains failures; it will not silently substitute an unpinned implementation."
if ($CreateEnvironments) {
    if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
        throw "Conda was not found. Install Conda or omit -CreateEnvironments and create the YAML environments manually."
    }
    Get-ChildItem -LiteralPath (Join-Path $root "external_methods\environments") -Filter "*.yaml" -File | ForEach-Object {
        conda env create --file $_.FullName
        if ($LASTEXITCODE -ne 0) { throw "Environment creation failed: $($_.Name)" }
    }
}
