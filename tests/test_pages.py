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
    assert "residual connection" in html
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
    assert "Expand, then contract" in html
    assert 'class="ff-network-svg"' in html
    assert "Feed-forward layer expands and contracts vector dimensions" in html
    assert "12 dimensions" in html
    assert 'id="feed-forward-code"' in html
    assert "FeedForwardLayer" in html
    assert 'nn.<span class="ff-fn">Linear</span>' in html
    assert 'llm/feed-forward-detail.js' not in html


def test_feed_forward_navigation_links_are_present(client):
    index_html = client.get("/").get_data(as_text=True)
    attention_html = client.get("/multi-head-attention").get_data(as_text=True)

    assert 'id="feed-forward"' in index_html
    assert 'href="/feed-forward"' in index_html
    assert 'href="/feed-forward"' in attention_html


def test_transformer_block_page_matches_source_and_completes_placeholders(client):
    response = client.get("/transformer-block")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "<h1>Transformer Block</h1>" in html
    assert "The core component of an LLM is the transformer block." in html
    assert 'href="/self-attention">self-attention</a>' in html
    assert 'href="/feed-forward">feed-forward network</a>' in html
    assert 'class="tf-diagram"' in html
    assert "The parts of a transformer block" in html
    assert "There are a couple of extra components here: LayerNorm and residual connections." in html
    assert "There is a built-in PyTorch implementation for this:" in html
    assert 'id="layer-norm-code"' in html
    assert 'id="transformer-without-residuals-code"' in html
    assert 'id="transformer-block-code"' in html
    assert "TransformerBlockWithoutResidualConnections" in html
    assert "MultiHeadAttention" in html
    assert "FeedForwardLayer" in html
    assert "TransformerBlock" in html


def test_transformer_block_navigation_link_is_present(client):
    html = client.get("/").get_data(as_text=True)

    assert 'id="transformer-block"' in html
    assert 'href="/transformer-block"' in html


def test_full_llm_page_matches_source_and_completes_placeholders(client):
    response = client.get("/full-llm")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "<h1>Full LLM</h1>" in html
    assert "A full language model turns a prompt into a probability distribution for the next token." in html
    assert "We have already encountered almost all of the components required to make an LLM." in html
    assert "project the final resulting vector into an array of logits" in html
    assert "The code now looks like this:" in html
    assert "The result of this is an array with the same size as the number of tokens" in html
    assert "We then use softmax to interpret this as a probability distribution" in html
    assert "One way of obtaining the next token at this point is to just use the most likely token:" in html
    assert "We can also make our LLM more variable by instead randomly sampling the probability distribution of tokens." in html
    assert 'id="full-llm-visual"' in html
    assert "Full language model pipeline" in html
    assert "Token + position embeddings" in html
    assert "Transformer block 1" in html
    assert "Vocabulary projection" in html
    assert 'id="full-llm-code"' in html
    assert "LanguageModel" in html
    assert "TransformerBlock" in html
    assert 'id="next-token-probabilities-code"' in html
    assert 'class="llm-fn">softmax</span>' in html
    assert 'id="greedy-decoding-code"' in html
    assert 'class="llm-fn">argmax</span>' in html
    assert 'id="sampling-code"' in html
    assert 'class="llm-fn">multinomial</span>' in html
    assert "TinyLanguageModel" not in html


def test_full_llm_navigation_link_is_present(client):
    html = client.get("/").get_data(as_text=True)

    assert 'id="llm-architecture"' in html
    assert 'href="/full-llm"' in html


def test_index_contains_open_weights_revised_copy_without_deep_dive(client):
    response = client.get("/")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert '<h2 id="open-weights">Open weights</h2>' in html
    source_paragraphs = (
        "Throughout an LLM, there are weights: in the matrices used for embeddings, the "
        "linear projections central to self-attention, the linear layers in the feed-forward "
        "network, and so on. Each of these components can contain thousands or millions of "
        "weights. Together, these weights allow the LLM to learn patterns from its training "
        "text and generate meaningful answers to prompts. Some frontier models contain "
        "hundreds of billions of weights.",
        "These weights are learned by training the model on a very large corpus of text: "
        "public web pages, code repositories, and many other sources. We’ll explore how "
        "this works later. Training an LLM this way can cost millions of dollars in hardware "
        "and electricity; for some frontier models, the figure can reach hundreds of millions.",
        "However, some companies and research teams make their model weights available for "
        "others to use. They publish files containing the trained weights of their LLMs, "
        "which other companies or researchers can load into their own instances of the model. "
        "Sites like Hugging Face host these files.",
    )
    for paragraph in source_paragraphs:
        assert f"<p>{paragraph}</p>" in html
    assert 'href="#open-weights"' in html
    assert 'href="/open-weights"' not in html


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
