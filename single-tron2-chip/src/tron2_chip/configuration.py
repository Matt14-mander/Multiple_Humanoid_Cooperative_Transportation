import json
from pathlib import Path


def load_config(path: Path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)

