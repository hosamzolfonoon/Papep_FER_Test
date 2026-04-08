#!/usr/bin/env bash
set -e

ENV_NAME=".FER"

echo "Creating virtual environment..."
python3 -m venv $ENV_NAME

echo "Activating environment..."
source $ENV_NAME/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

echo "Installing requirements..."
pip install -r requirements.txt

echo ""
echo "Setup completed!"
echo "Run later with:"
echo "source .venv/bin/activate"