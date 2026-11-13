#!/bin/bash
set -e
cd "$(dirname "$0")"
source .venv/bin/activate
python run_word_overlap.py
