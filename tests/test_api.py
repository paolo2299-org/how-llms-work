import pytest
from unittest.mock import MagicMock, patch

import numpy as np

from app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def _fake_glove(*words):
    rng = np.random.default_rng(42)
    return {w: rng.random(50).astype(np.float32) for w in words}


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


def test_tokenize_returns_tokens_and_ids(client):
    mock_enc = MagicMock()
    mock_enc.encode.return_value = [15496, 995]
    mock_enc.decode_single_token_bytes.side_effect = lambda t: {
        15496: b"Hello",
        995: b" world",
    }[t]

    with patch("app.routes.api.tiktoken.get_encoding", return_value=mock_enc):
        response = client.get("/api/tokenize?text=Hello world")

    assert response.status_code == 200
    data = response.get_json()
    assert data["tokens"] == ["Hello", " world"]
    assert data["token_ids"] == [15496, 995]


def test_tokenize_missing_text_returns_400(client):
    response = client.get("/api/tokenize")
    assert response.status_code == 400


def test_tokenize_empty_string(client):
    mock_enc = MagicMock()
    mock_enc.encode.return_value = []

    with patch("app.routes.api.tiktoken.get_encoding", return_value=mock_enc):
        response = client.get("/api/tokenize?text=")

    assert response.status_code == 200
    data = response.get_json()
    assert data["tokens"] == []
    assert data["token_ids"] == []


def test_embed_returns_3d_points(client):
    fake = _fake_glove("king", "queen", "dog")
    with patch.dict("app.glove.glove_vectors", fake, clear=True):
        response = client.post("/api/embed", json={"words": ["king", "queen", "dog"]})
    assert response.status_code == 200
    data = response.get_json()
    assert len(data["points"]) == 3
    assert data["unknown"] == []
    point = data["points"][0]
    assert all(k in point for k in ["word", "x", "y", "z"])


def test_embed_unknown_words(client):
    with patch.dict("app.glove.glove_vectors", {}, clear=True):
        response = client.post("/api/embed", json={"words": ["zzznonsense"]})
    assert response.status_code == 200
    data = response.get_json()
    assert data["points"] == []
    assert "zzznonsense" in data["unknown"]


def test_embed_mixed_known_unknown(client):
    fake = _fake_glove("cat", "dog")
    with patch.dict("app.glove.glove_vectors", fake, clear=True):
        response = client.post("/api/embed", json={"words": ["cat", "zzznonsense", "dog"]})
    assert response.status_code == 200
    data = response.get_json()
    words_returned = [p["word"] for p in data["points"]]
    assert "cat" in words_returned
    assert "dog" in words_returned
    assert "zzznonsense" in data["unknown"]


def test_embed_single_word(client):
    fake = _fake_glove("hello")
    with patch.dict("app.glove.glove_vectors", fake, clear=True):
        response = client.post("/api/embed", json={"words": ["hello"]})
    assert response.status_code == 200
    data = response.get_json()
    assert len(data["points"]) == 1
    p = data["points"][0]
    assert p["word"] == "hello"
    assert p["x"] == 0.0 and p["y"] == 0.0 and p["z"] == 0.0


def test_embed_two_words(client):
    fake = _fake_glove("hello", "world")
    with patch.dict("app.glove.glove_vectors", fake, clear=True):
        response = client.post("/api/embed", json={"words": ["hello", "world"]})
    assert response.status_code == 200
    data = response.get_json()
    assert len(data["points"]) == 2
    for p in data["points"]:
        assert p["y"] == 0.0 and p["z"] == 0.0


def test_embed_empty_word_list(client):
    with patch.dict("app.glove.glove_vectors", {}, clear=True):
        response = client.post("/api/embed", json={"words": []})
    assert response.status_code == 200
    data = response.get_json()
    assert data["points"] == []
    assert data["unknown"] == []
