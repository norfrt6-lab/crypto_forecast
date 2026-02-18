#!/bin/bash
set -e

echo "Running linters..."
flake8 src/ || true
mypy src/ || true
ruff check src/ || true
echo "Lint complete."
