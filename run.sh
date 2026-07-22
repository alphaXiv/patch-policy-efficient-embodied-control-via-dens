#!/usr/bin/env bash
set -euo pipefail

python -m pip install --disable-pip-version-check -q -r requirements.txt
python reproduce.py --config config.json
