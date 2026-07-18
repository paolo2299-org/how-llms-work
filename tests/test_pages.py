import pytest

from app import create_app


@pytest.fixture
def client():
    app = create_app(load_glove=False)
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


def test_index_contains_transformer_overviews(client):
    response = client.get("/")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert 'id="transformer-block"' in html
    assert "LayerNorm" in html
    assert "shortcut connection" in html
    assert 'id="llm-architecture"' in html
    assert "Transformer block 1" in html
    assert "Output head" in html
    assert "Next token" in html
