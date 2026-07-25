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
    assert 'href="/full-llm"' in html


def test_index_table_of_contents_links_overviews_and_detail_pages(client):
    html = client.get("/").get_data(as_text=True)

    assert 'id="toc-heading"' in html
    assert 'href="#tokenisation"' in html
    assert 'href="/tokenisation"' in html
    assert html.index('href="#tokenisation"') < html.index('href="/tokenisation"')
    assert 'href="#token-embeddings"' in html
    assert 'href="/token-embeddings"' in html
    assert 'href="#self-attention"' in html
    assert 'href="/self-attention"' in html
    assert 'href="/multi-head-attention"' in html
    assert 'href="#feed-forward"' in html
    assert 'href="/feed-forward"' in html
    assert 'href="#transformer-block"' in html
    assert 'href="/transformer-block"' in html
    assert 'href="#llm-architecture"' in html
    assert 'href="/full-llm"' in html


def test_feed_forward_page_matches_source_and_completes_placeholders(client):
    response = client.get("/feed-forward")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "<h1>Feed-Forward Layer</h1>" in html
    assert "helps an LLM to <em>learn features</em>" in html
    assert "This phrase expresses disagreement." in html
    assert "Projects the result back to the original vector size" in html
    assert 'id="feed-forward-visual"' in html
    assert "Widen, activate, then narrow" in html
    assert 'aria-labelledby="ff-visual-title"' in html
    assert 'id="feed-forward-code"' in html
    assert "FeedForwardLayer" in html
    assert 'nn.<span class="ff-fn">Linear</span>' in html
    assert 'llm/feed-forward-detail.js' in html


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
    assert "Transformer Block" in html
    assert "Pre-normalised transformer block" in html
    assert "LayerNorm 1" in html
    assert "first residual connection" in html
    assert 'id="transformer-block-class"' in html
    assert "TransformerBlock" in html


def test_transformer_block_navigation_link_is_present(client):
    html = client.get("/").get_data(as_text=True)

    assert 'id="transformer-block"' in html
    assert 'href="/transformer-block"' in html


def test_full_llm_page_contains_worked_example(client):
    response = client.get("/full-llm")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Full LLM" in html
    assert "Decoder-only LLM forward pass" in html
    assert "Token + position embeddings" in html
    assert "Transformer block 1" in html
    assert "softmax" in html
    assert 'id="full-llm-class"' in html
    assert "TinyLanguageModel" in html


def test_full_llm_navigation_link_is_present(client):
    html = client.get("/").get_data(as_text=True)

    assert 'id="llm-architecture"' in html
    assert 'href="/full-llm"' in html


def test_tokenisation_page_contains_worked_example(client):
    response = client.get("/tokenisation")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "<h1>Tokenisation</h1>" in html
    assert "Word-based tokeniser" in html
    assert "How LLMs actually do it" in html
    assert "Byte Pair Encoding (BPE)" in html
    assert "tok-figure-title" not in html
    assert 'id="tokeniser-function"' in html
    assert "tokenise" in html


def test_tokenisation_navigation_link_is_present(client):
    html = client.get("/").get_data(as_text=True)

    assert 'id="tokenisation"' in html
    assert 'href="/tokenisation"' in html


def test_token_embeddings_page_contains_worked_example(client):
    response = client.get("/token-embeddings")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "<h1>Token Embeddings</h1>" in html
    assert 'href="/tokenisation">tokenisation deep dive</a>' in html
    assert "There are many reasons why vectors are the natural objects" in html
    assert "<h2>The embedding matrix</h2>" in html
    assert "first row is the vector that the token with ID 0" in html
    assert "The vertical dots stand in for all the other rows" in html
    assert "<h2>Positional embeddings</h2>" in html
    assert 'id="token-embedding-code"' in html
    assert 'id="position-embedding-code"' in html
    assert "token_embedding_matrix" in html
    assert "position_embedding_matrix" in html


def test_token_embeddings_navigation_link_is_present(client):
    html = client.get("/").get_data(as_text=True)

    assert 'id="token-embeddings"' in html
    assert 'href="/token-embeddings"' in html
