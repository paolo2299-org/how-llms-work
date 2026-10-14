"""Missing assets have a deliberate development fallback, explicit mistakes fail."""

from pathlib import Path

import pytest

from scripts import dev_app


def test_explicit_glove_path_is_resolved_from_worktree(tmp_path, monkeypatch):
    (tmp_path / "vectors.txt").write_text("word 0.1 0.2\n")
    monkeypatch.setenv("DEV_GLOVE_PATH", "vectors.txt")
    assert dev_app.glove_file(tmp_path) == tmp_path / "vectors.txt"


def test_missing_explicit_vectors_are_reported(tmp_path, monkeypatch):
    monkeypatch.setenv("DEV_GLOVE_PATH", str(tmp_path / "missing"))
    with pytest.raises(ValueError, match="GloVe file not found"):
        dev_app.glove_file(tmp_path)


def test_no_vectors_allow_page_development(monkeypatch):
    monkeypatch.setenv("FLASK_ENV", "development")
    monkeypatch.setattr(dev_app, "load_dotenv", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(dev_app, "glove_file", lambda _root: None)
    monkeypatch.setattr(dev_app, "make_server", lambda _host, _port, app, **_kwargs: app)
    app, assets = dev_app.create_server(Path(__file__).resolve().parents[1], "127.0.0.1")
    assert app.test_client().get("/").status_code == 200
    assert app.test_client().get("/health").status_code == 200
    assert assets["glove_file"] is None
