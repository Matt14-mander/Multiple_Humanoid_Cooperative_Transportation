import json
from pathlib import Path
from typing import Any, Dict

from .paths import DEFAULT_CONFIG


def load_config(path: Path = DEFAULT_CONFIG) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)

