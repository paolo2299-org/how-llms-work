import numpy as np

glove_vectors: dict = {}


def load_glove_vectors(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            glove_vectors[parts[0]] = np.array(parts[1:], dtype=np.float32)
