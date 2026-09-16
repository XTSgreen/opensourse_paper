#!/usr/bin/env bash
set -euo pipefail
if [[ "${1:-}" == "--download-raw" ]]; then
  python data/download/download_public_data.py
else
  python data/download/download_public_data.py --dry-run
  echo "Frozen processed panels are packaged locally. Pass --download-raw to fetch the optional full GEO archives."
fi
python scripts/reproduce.py prepare
