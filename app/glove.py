from threading import Lock
import numpy as np
from flask import current_app

glove_vectors: dict = {}
glove_vectors_loaded = False
glove_vectors_lock = Lock()


def load_glove_vectors(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                glove_vectors[parts[0]] = np.array(parts[1:], dtype=np.float32)
    except FileNotFoundError:
        pass


def ensure_glove_vectors_loaded():
    global glove_vectors_loaded

    if glove_vectors_loaded or glove_vectors:
        glove_vectors_loaded = True
        return

    with glove_vectors_lock:
        if glove_vectors_loaded or glove_vectors:
            glove_vectors_loaded = True
            return

        path = current_app.config["GLOVE_PATH"]
        load_glove_vectors(path)
        glove_vectors_loaded = True
