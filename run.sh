#!/usr/bin/env bash
deactivate &> /dev/null
source .python_env_lnx/bin/activate
python -O gui.py
