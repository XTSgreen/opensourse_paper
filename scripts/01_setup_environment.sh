#!/usr/bin/env bash
set -euo pipefail
python -m pip install -e '.[test]'
python -m pytest -q
