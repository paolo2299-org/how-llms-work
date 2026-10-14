"""Use a shared GloVe file for reading, or run page-only development without it."""

import os
from pathlib import Path
import subprocess

from dotenv import load_dotenv
from werkzeug.serving import make_server


def glove_file(root: Path) -> Path | None:
    explicit = os.environ.get("DEV_GLOVE_PATH") or os.environ.get("GLOVE_PATH")
    if explicit:
        path = Path(explicit).expanduser()
        path = (root / path).resolve() if not path.is_absolute() else path.resolve()
        if not path.is_file():
            raise ValueError(f"GloVe file not found: {path}. Set DEV_GLOVE_PATH to a native host path.")
        return path
    common = Path(subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "--path-format=absolute", "--git-common-dir"], text=True,
    ).strip())
    candidates = (root / "data/glove.6B.50d.txt", common.parent / "data/glove.6B.50d.txt",
                  Path("/srv/how-llms-work/data/glove.6B.50d.txt"))
    return next((path for path in candidates if path.is_file()), None)


def create_server(root: Path, host: str):
    load_dotenv(root / ".env", override=False)
    glove = glove_file(root)
    if glove is not None:
        os.environ["GLOVE_PATH"] = str(glove)
    else:
        print("No GloVe file found; pages/tokenisation work, embedding results are unavailable.", flush=True)
    from app import create_app

    return make_server(host, 0, create_app(load_glove=glove is not None), threaded=True), {
        "glove_file": str(glove) if glove else None,
    }
