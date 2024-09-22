#!/usr/bin/env bash
python -m venv .python_env_lnx
source .python_env_lnx/bin/activate
pip install pip -U
pip install uv -U
uv pip install Pillow numpy stable_baselines3 sb3_contrib
cd models
./getmodels.sh
cd ../
