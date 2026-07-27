"""Inspect and later host the original encoder/policy ONNX sessions."""

import argparse
from pathlib import Path
from typing import Dict, List

from .paths import POLICY_DIR


def inspect_model(path: Path) -> Dict[str, List[Dict[str, object]]]:
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise RuntimeError(
            "onnxruntime is optional; install the 'policy' extra first"
        ) from error
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return {
        "inputs": [
            {"name": item.name, "shape": item.shape, "type": item.type}
            for item in session.get_inputs()
        ],
        "outputs": [
            {"name": item.name, "shape": item.shape, "type": item.type}
            for item in session.get_outputs()
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-dir", type=Path, default=POLICY_DIR)
    args = parser.parse_args()
    for filename in ("encoder.onnx", "policy.onnx"):
        path = args.policy_dir / filename
        print("{}: {}".format(filename, inspect_model(path)))


if __name__ == "__main__":
    main()

