#!/bin/bash

# Exit script if any command fails
set -e

echo "syntax on" >> ~/.vimrc
mkdir -p data
mkdir -p predictions
python3 -m venv myenv
source myenv/bin/activate  # On Windows, use `myenv\Scripts\activate`
pip install -r requirements.txt

