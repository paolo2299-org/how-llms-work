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


def test_feed_forward_page_contains_worked_example(client):
    response = client.get("/feed-forward")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Feed-Forward Layer, Step by Step" in html
    assert "shape (5, 6)" in html
    assert "shape (5, 12)" in html
    assert "GELU" in html
    assert 'id="feed-forward-class"' in html
    assert "FeedForward" in html
    assert "Feed-forward tensor shape flow" in html


def test_feed_forward_navigation_links_are_present(client):
    index_html = client.get("/").get_data(as_text=True)
    attention_html = client.get("/multi-head-attention").get_data(as_text=True)

    assert 'id="feed-forward"' in index_html
    assert 'href="/feed-forward"' in index_html
    assert 'href="/feed-forward"' in attention_html


def test_transformer_block_page_contains_worked_example(client):
    response = client.get("/transformer-block")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Transformer Block, Step by Step" in html
    assert "Pre-normalised transformer block" in html
    assert "LayerNorm 1" in html
    assert "first residual connection" in html
    assert 'id="transformer-block-class"' in html
    assert "TransformerBlock" in html


def test_transformer_block_navigation_link_is_present(client):
    html = client.get("/").get_data(as_text=True)

    assert 'id="transformer-block"' in html
    assert 'href="/transformer-block"' in html
