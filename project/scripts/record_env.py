import subprocess
import sys

from musichg.utils import env_info, save_json

info = env_info()
info["pip_freeze"] = subprocess.check_output([sys.executable, "-m", "pip", "freeze"]).decode().splitlines()
save_json(info, "runs/env.json")
print("wrote runs/env.json")
