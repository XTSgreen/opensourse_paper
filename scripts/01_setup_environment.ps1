Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
python -m pip install -e ".[test]"
python -m pytest -q
