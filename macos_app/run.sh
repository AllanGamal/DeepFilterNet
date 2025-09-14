#!/bin/bash
# Simple script to run the DeepFilterNet macOS app

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Go to the repository root
REPO_ROOT="$(dirname "$SCRIPT_DIR")"

# Activate the virtual environment
source "$REPO_ROOT/venv_app/bin/activate"

# Change to the macos_app directory
cd "$SCRIPT_DIR"

# Run the app
python3 app.py
