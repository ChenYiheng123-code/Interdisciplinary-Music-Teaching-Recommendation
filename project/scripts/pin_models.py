
import datetime
import json

import yaml
from huggingface_hub import HfApi

M = yaml.safe_load(open("models.yaml"))
api = HfApi(); lock = {"resolved_at_utc": datetime.datetime.utcnow().isoformat() + "Z"}
for k, v in M.items():
    info = api.model_info(v["name"])
    lock[k] = dict(name=v["name"], revision=info.sha)
    print(k, v["name"], info.sha)
json.dump(lock, open("models.lock.json", "w"), indent=2)
