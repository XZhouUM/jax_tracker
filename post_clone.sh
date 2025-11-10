#!/bin/bash
set -e

echo "Setting up Git hooks..."

git config core.hooksPath .githooks

if [ -f .githooks/pre-commit ]; then
  chmod +x .githooks/pre-commit
  echo "pre-commit hook is installed and executable."
else
  echo "Warning: .githooks/pre-commit not found!"
fi
