import json
from pathlib import Path
from datetime import datetime

def make_run_dir(task: str, exp: str) -> Path:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path("runs") / task / f"{exp}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def log_json(path, payload: dict):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
