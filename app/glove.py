from threading import Lock
import numpy as np
from flask import current_app

glove_vectors: dict = {}
glove_vectors_loaded = False
glove_vectors_lock = Lock()
glove_load_error: str | None = None


def load_glove_vectors(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            glove_vectors[parts[0]] = np.array(parts[1:], dtype=np.float32)


def ensure_glove_vectors_loaded():
    global glove_vectors_loaded, glove_load_error

    if glove_vectors_loaded:
        return

    with glove_vectors_lock:
        if glove_vectors_loaded:
            return

        path = current_app.config["GLOVE_PATH"]
        try:
            load_glove_vectors(path)
        except FileNotFoundError:
            glove_load_error = f"GloVe data file not found: {path}"
            current_app.logger.error(glove_load_error)
        except Exception as e:
            glove_load_error = f"Failed to load GloVe data from {path}: {e}"
            current_app.logger.error(glove_load_error)
        finally:
            glove_vectors_loaded = True
