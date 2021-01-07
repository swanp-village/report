#!/bin/bash
# Install or update needed software
sudo apt-get update
sudo apt-get install -yq git python3.7 python3-venv

# Fetch source code
git clone https://github.com/nimiusrd/MRR.git
cd MRR || exit

# Python environment setup
python3 -m venv .venv
# shellcheck disable=SC1091
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
