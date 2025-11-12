#!/bin/bash
rm -rf build/ dist/ match_attention.egg-info/ __pycache__
python setup.py clean
pip install .