import yaml
import subprocess
import os
import fnmatch


exclude_list = ["config/exclude.yaml",]

config_path = "config"
config_files = [
    os.path.join(config_path, f) for f in os.listdir(config_path) if f.endswith(".yaml")
]
config_files = [
    f
    for f in config_files
    if not any(fnmatch.fnmatch(f, pattern) for pattern in exclude_list)
]

for config_file in config_files:
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    command = ["python", "train/trainer.py", "--config", config_file]
    p = subprocess.Popen(command)
    p.wait()
