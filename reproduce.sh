#!/usr/bin/env sh
# Regenerate every number, check and figure of the paper.
set -e
cd "$(dirname "$0")/code"
python make_numbers.py
python run_checks.py
python check_small_time.py
python make_figures.py --recompute
python make_hero.py
python make_hero_labels.py
python make_web_data.py
