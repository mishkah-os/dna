#!/usr/bin/env bash
set -euo pipefail

cd /home/huss/projects/dna
if [ -f .env.production ]; then
  set -a
  . ./.env.production
  set +a
fi

exec ./.venv/bin/python app.py
