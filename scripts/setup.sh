#!/bin/bash
set -e

echo "Setting up development environment..."

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
pip install -r requirements-dev.txt 2>/dev/null || true

pre-commit install 2>/dev/null || true

echo "Setup complete. Activate with: source .venv/bin/activate"
