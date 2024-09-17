#!/usr/bin/env pwsh
python -m venv .python_env_win
.python_env_win\Scripts\activate.ps1
pip install pip -U
pip install uv -U
uv pip install Pillow numpy stable_baselines3 sb3_contrib
python -O solver.py
