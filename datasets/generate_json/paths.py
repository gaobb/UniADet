import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def dataset_dir(*parts: str) -> str:
    root = Path(os.environ.get("UNIADET_DATA", _REPO_ROOT / "data"))
    return str(root.joinpath(*parts))
