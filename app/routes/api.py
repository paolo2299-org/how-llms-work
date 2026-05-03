from flask import Blueprint, jsonify, request
import numpy as np
from sklearn.decomposition import PCA
import tiktoken

from app.glove import ensure_glove_vectors_loaded, glove_vectors

api_bp = Blueprint("api", __name__)


@api_bp.route("/health")
def health():
    return jsonify({"status": "ok"})


@api_bp.route("/api/tokenize")
def tokenize():
    text = request.args.get("text")
    if text is None:
        return jsonify({"error": "missing required query parameter: text"}), 400
    enc = tiktoken.get_encoding("cl100k_base")
    token_ids = enc.encode(text)
    tokens = [enc.decode_single_token_bytes(t).decode("utf-8", errors="replace") for t in token_ids]
    return jsonify({"tokens": tokens, "token_ids": list(token_ids)})


@api_bp.route("/api/embed", methods=["POST"])
def embed():
    body = request.get_json(force=True, silent=True) or {}
    words = body.get("words", [])

    ensure_glove_vectors_loaded()

    known, unknown, vectors = [], [], []
    for word in words:
        w = word.lower()
        if w in glove_vectors:
            known.append(word)
            vectors.append(glove_vectors[w])
        else:
            unknown.append(word)

    if not known:
        return jsonify({"points": [], "unknown": unknown})

    arr = np.array(vectors)
    n = len(known)

    if n == 1:
        coords = np.zeros((1, 3))
    elif n == 2:
        pca = PCA(n_components=1)
        r = pca.fit_transform(arr)
        coords = np.hstack([r, np.zeros((2, 2))])
    else:
        n_components = min(3, n)
        pca = PCA(n_components=n_components)
        r = pca.fit_transform(arr)
        pad = np.zeros((n, 3 - r.shape[1]))
        coords = np.hstack([r, pad])

    points = [
        {"word": known[i], "x": float(coords[i, 0]), "y": float(coords[i, 1]), "z": float(coords[i, 2])}
        for i in range(n)
    ]
    return jsonify({"points": points, "unknown": unknown})
