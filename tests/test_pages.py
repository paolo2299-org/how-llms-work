from pathlib import Path

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


def test_index_introduction_describes_the_learning_path(client):
    html = client.get("/").get_data(as_text=True)

    assert (
        "This page gives a high level overview of all of the core concepts that "
        "underpin the tech behind LLMs. Each section has one or more deep dives, "
        "where you’ll piece together the complete code for a fully functional LLM."
        in html
    )
    assert (
        "No specialist knowledge is required to read this overview page, but to get "
        "the most out of the deep dive pages some familiarity with python and neural "
        "networks is recommended."
        in html
    )
    assert "Here are a couple of excellent resources for this:" in html
    assert (
        'href="https://victorzhou.com/blog/intro-to-neural-networks/">An introduction to neural networks</a>'
        in html
    )
    assert (
        'href="https://www.python.org/about/gettingstarted/">Python for beginners</a>'
        in html
    )
    assert (
        'href="https://github.com/paolo2299-org/how-llms-work/tree/main/code">here</a>'
        in html
    )


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
    assert 'href="#open-weights"' in html
    assert 'href="/open-weights"' in html
    assert 'href="#pre-training"' in html
    assert 'href="/pre-training"' in html
    assert 'href="/pre-training/model-additions"' in html
    assert 'href="/pre-training/weight-optimisation"' in html
    assert 'href="/pre-training/full-loop"' in html
    assert 'href="#post-training"' in html
    assert 'href="/fine-tuning"' in html
    assert 'href="#summary"' in html


@pytest.mark.parametrize(
    ("section_id", "next_section_id", "destination"),
    [
        ("tokenisation", "token-embeddings", "/tokenisation"),
        ("token-embeddings", "self-attention", "/token-embeddings"),
        ("self-attention", "feed-forward", "/self-attention"),
        ("feed-forward", "transformer-block", "/feed-forward"),
        ("transformer-block", "llm-architecture", "/transformer-block"),
        ("llm-architecture", "open-weights", "/full-llm"),
        ("open-weights", "pre-training", "/open-weights"),
        ("pre-training", "post-training", "/pre-training"),
        ("post-training", "summary", "/fine-tuning"),
    ],
)
def test_index_sections_end_with_their_first_deep_dive_link(
    client, section_id, next_section_id, destination
):
    html = client.get("/").get_data(as_text=True)
    section = html[html.index(f'id="{section_id}"') : html.index(f'id="{next_section_id}"')]

    assert f'href="{destination}"' in section


@pytest.mark.parametrize(
    ("path", "previous_destination", "next_destination"),
    [
        ("/tokenisation", "/#tokenisation", "/#token-embeddings"),
        ("/token-embeddings", "/#token-embeddings", "/#self-attention"),
        ("/self-attention", "/#self-attention", "/multi-head-attention"),
        ("/multi-head-attention", "/self-attention", "/#feed-forward"),
        ("/feed-forward", "/#feed-forward", "/#transformer-block"),
        ("/transformer-block", "/#transformer-block", "/#llm-architecture"),
        ("/full-llm", "/#llm-architecture", "/#open-weights"),
        ("/open-weights", "/#open-weights", "/#pre-training"),
        ("/pre-training", "/#pre-training", "/pre-training/model-additions"),
        (
            "/pre-training/model-additions",
            "/pre-training",
            "/pre-training/weight-optimisation",
        ),
        (
            "/pre-training/weight-optimisation",
            "/pre-training/model-additions",
            "/pre-training/full-loop",
        ),
        (
            "/pre-training/full-loop",
            "/pre-training/weight-optimisation",
            "/#post-training",
        ),
        ("/fine-tuning", "/#post-training", "/#summary"),
    ],
)
def test_deep_dives_have_logical_navigation_at_the_top_and_bottom(
    client, path, previous_destination, next_destination
):
    html = client.get(path).get_data(as_text=True)

    assert html.count('aria-label="Deep dive navigation"') == 2
    assert html.count(f'href="{previous_destination}"') >= 2
    assert html.count(f'href="{next_destination}"') >= 2


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


def test_index_contains_open_weights_revised_copy_with_deep_dive(client):
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
    assert 'href="/open-weights"' in html


def test_open_weights_page_matches_source_and_completes_placeholder(client):
    response = client.get("/open-weights")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "<h1>Open weights</h1>" in html
    source_paragraphs = (
        "We’ll now see how to load a weights file into our model. For this we'll use the "
        "published weights for the GPT-2 model, which is an older model (2019), but it is "
        "small and simple enough to work with on a modest laptop.",
        "An important thing to note when loading open weights using your own LLM code is "
        "that your LLM needs to be compatible with the architecture and parameter shapes "
        "of the LLM that produced the weights. There are many choices you can make when "
        "building an LLM: the number of transformer layers, projection dimensions, choice "
        "of activation functions, and many more. In our case, we have carefully constructed "
        "our LLM to be compatible with GPT-2 weights.",
        "Now the code to actually load the weights into our model looks as follows:",
    )
    for paragraph in source_paragraphs:
        assert f"<p>{paragraph}</p>" in html

    assert 'id="weight-loading-overview"' not in html
    assert "From a checkpoint file to a ready model" not in html
    assert 'id="load-gpt2-code"' in html
    assert "load_gpt2_small" in html
    assert "weights_only=" in html
    assert 'id="translate-weights-code"' in html
    assert "matching each dictionary entry to a model parameter by its exact name" in html
    assert "Calling <code>load_state_dict</code> with the original names" in html
    assert "embedding.token_embedding.weight" in html
    assert "transformer_blocks.0.attention.W_query.weight" in html
    assert "strict=" in html
    assert 'id="generate-with-weights-code"' in html
    assert "Work in progress" not in html


def test_index_contains_post_training_overview_and_completed_summary(client):
    html = client.get("/").get_data(as_text=True)

    assert '<h2 id="pre-training">Pre-training</h2>' in html
    assert '<h2 id="post-training">Post-training</h2>' in html
    assert "the model has learned broad patterns in language" in html
    assert "not yet the kind of chatbot people commonly associate with an LLM" in html
    assert "preference optimisation using human or model feedback" in html
    assert "Instruction fine-tuning: in depth →" in html
    assert '<h2 id="summary">Summary</h2>' in html
    assert "We’ve now learned all of the core concepts of an LLM" in html
    assert (
        'href="https://github.com/paolo2299-org/how-llms-work/tree/main/code/'
        'llm_inference_only">A slightly simplified LLM</a>' in html
    )
    assert (
        'href="https://github.com/paolo2299-org/how-llms-work/tree/main/code/llm">'
        "A modified version of the above LLM</a>" in html
    )
    for difference in ("Scale:", "Architecture:", "Training:", "Post-training:", "Inference:"):
        assert f"<strong>{difference}</strong>" in html
    assert "Work in progress" not in html


def test_index_in_depth_calls_to_action_contain_only_links(client):
    html = client.get("/").get_data(as_text=True)

    in_depth_links = (
        ('/tokenisation', "Tokenisation: in depth →"),
        ('/token-embeddings', "Token embeddings: in depth →"),
        ('/self-attention', "Self-attention: in depth →"),
        ('/feed-forward', "Feed-forward layer: in depth →"),
        ('/transformer-block', "Transformer block: in depth →"),
        ('/full-llm', "Full LLM: in depth →"),
        ('/open-weights', "Open weights: in depth →"),
        ('/fine-tuning', "Instruction fine-tuning: in depth →"),
    )
    for href, label in in_depth_links:
        assert f'<p><a href="{href}">{label}</a></p>' in html

    assert "Want to inspect the mechanism?" not in html
    assert "Want to follow the tensors?" not in html
    assert "Want to see the actual mechanism?" not in html
    assert "Want to follow the numbers?" not in html
    assert "Want to see how those wrappers fit" not in html
    assert "Want to connect all the parts?" not in html


def test_index_pre_training_section_matches_source_and_completes_placeholders(client):
    response = client.get("/")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert '<h2 id="pre-training">Pre-training</h2>' in html
    source_markup = (
        'We talked above about how models have millions or even billions of weights. The LLM we finally managed to build in <a href="/full-llm">The Full LLM</a> section had random weights, and if you go ahead and generate text with it, you will get back random gibberish:',
        'We then showed how to load weights for a very small but trained model in the <a href="/open-weights">Open Weights</a> section, which produces more coherent (if a little uninspired!) results:',
        "To get from our random weights to weights that will produce coherent sentences, we use a process called <em>pre-training</em>.",
        "This involves the following steps:",
        "Build a large dataset of text written by humans from internet-based sources, such as websites, books, articles, academic papers, code repositories, etc., and convert it to tokens.",
        "Initialize the LLM with random weights",
        "Feed “chunks” of the input text into the model. One such chunk might be “the cat sat on the mat”.",
        "Our model now produces next-token probabilities that tell us how well it would have predicted the correct sentence from the incomplete sentence:",
        "In fact, the architecture of our model means that in a single pass it can actually simultaneously calculate next token probabilities for all incomplete versions of our sentence:",
        "We update the weights based on these observed probabilities. If the model gave us high probabilities, that means that it did well at predicting our text, and so our weights are working well, and we don’t need to adjust them much. If the probabilities are low, then the weights are not working well, and we need to adjust them more. We use a process called <em>backpropagation</em> to calculate how changing each weight would affect the model’s prediction error, and a separate optimisation process uses this information to adjust the weights.",
        "We repeat steps 3 to 5 over and over again, iterating over our dataset, until further training produces too little improvement to justify the additional computing cost.",
    )
    for source_text in source_markup:
        assert source_text in html

    assert "romancersurface crimesAud dissectiffs abol" in html
    assert "floor, and the cat was sitting on the floor" in html
    assert html.count('class="pt-probabilities"') == 2
    assert "The probability of “on” is 0.01" in html
    assert "The probability of “the” is 0.06" in html
    deep_dive_links = (
        ('/pre-training', "In depth: preparing the inputs"),
        ('/pre-training/model-additions', "In depth: model additions"),
        ('/pre-training/weight-optimisation', "In depth: weight optimisation"),
        ('/pre-training/full-loop', "In depth: the full pre-training loop"),
    )
    for href, label in deep_dive_links:
        assert f'href="{href}">{label}</a>' in html


def test_pre_training_inputs_page_matches_training_implementation(client):
    response = client.get("/pre-training")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "<h1>Pre-training: preparing the inputs</h1>" in html
    assert 'href="/tokenisation">tokeniser</a>' in html
    assert "extract a collection of inputs and targets" in html
    assert "Winston Smith, his chin nuzzled into his breast" in html
    assert "George Orwell, <cite>Nineteen Eighty-Four</cite>" in html
    assert 'class="ptd-shift-figure"' in html
    assert 'aria-label="Input and target pair 1"' in html
    assert "Two input/target pairs made from the example text" in html
    assert "&#x2420;was" in html
    assert (
        'href="https://github.com/paolo2299-org/how-llms-work/tree/main/code/llm"'
        in html
    )
    assert 'id="next-token-dataset-code"' in html
    assert "class NextTokenDataset(Dataset):" in html
    assert "len(self.token_ids) - sequence_length" in html
    assert "inputs = self.token_ids[start:stop]" in html
    assert "targets = self.token_ids[start + 1 : stop + 1]" in html
    assert "For a given input and target pair" in html
    assert "At position 0 the model is asked to predict" in html
    assert "Why is one extra token required?" not in html
    assert 'id="create-data-loader-code"' in html
    assert "DataLoader(dataset, batch_size=batch_size, shuffle=True)" in html
    assert "The dataset returns one input/target pair at a time" in html
    assert "Shapes created by the dataset and data loader" not in html
    assert (
        "here just randomises the order in which each input/target pair is added to a batch"
        in html
    )
    assert "the order that each input/target pair" not in html
    assert "Work in progress" not in html
    assert 'href="/#pre-training"' in html
    assert 'href="/pre-training/model-additions"' in html

    source = Path("code/llm/pretrain.py").read_text(encoding="utf-8")
    for code_line in (
        "len(self.token_ids) - sequence_length",
        "inputs = self.token_ids[start:stop]",
        "targets = self.token_ids[start + 1 : stop + 1]",
        "return DataLoader(dataset, batch_size=batch_size, shuffle=True)",
    ):
        assert code_line in source


def test_pre_training_model_additions_matches_source_and_completes_placeholders(client):
    response = client.get("/pre-training/model-additions")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "<h1>Pre-training: model additions</h1>" in html
    assert "The LLM implementation we have built so far" in html
    assert (
        'href="https://github.com/paolo2299-org/how-llms-work/blob/main/'
        'code/llm_inference_only/language_model.py"'
        in html
    )
    assert "Batching (passing multiple inputs in at the same time)" in html
    assert "Producing next token predictions for every input token" in html
    assert "<h2>Batching</h2>" in html
    assert 'id="single-sequence-example"' in html
    assert "[“It”, “ was”, “ a”, “ bright”, “ cold”]" in html
    assert 'id="batch-example"' in html
    assert "A batch containing three sequences of five tokens" in html
    assert "107nPkg3n-CnKQSa2JI4Zk8B-UB71ZBhqqwmXq_Lazfs" in html
    assert 'id="batched-token-ids-code"' in html
    assert "[1026,   373,  257, 6016,  4692]" in html
    assert 'id="batched-embedding-code"' in html
    assert "token_ids must have shape (batch_size, sequence_length)" in html
    assert "# PyTorch broadcasts the position" not in html
    assert 'id="embedding-output-visual"' in html
    assert "shape <code>(3, 5, model_dim)</code>" in html
    assert 'id="batched-attention-code"' in html
    assert "batch_size, num_tokens, _ = x.shape" in html
    assert ").transpose(1, 2)" in html
    assert "The batch items never attend to one another" in html
    assert "Next-token probabilities observed for every incomplete version" in html
    for probability_example in (
        "The probability of “cat” is 0.004",
        "The probability of “sat” is 0.03",
        "The probability of “on” is 0.01",
        "The probability of “the” is 0.06",
        "The probability of “mat” is 0.02",
    ):
        assert probability_example in html
    assert 'id="all-position-logits-code"' in html
    assert "return self.vocabulary_projection(x)" in html
    assert 'id="batched-generation-code"' in html
    assert "next_token_logits = all_logits[0, -1]" in html
    assert "This lets one model implementation support both batched training" in html
    assert 'href="/pre-training/weight-optimisation"' in html
    assert "<show " not in html
    assert "<insert " not in html
    assert "<use " not in html
    assert "<repeat " not in html

    source_files = {
        "embedding": Path("code/llm/token_embedding.py").read_text(encoding="utf-8"),
        "attention": Path("code/llm/self_attention.py").read_text(encoding="utf-8"),
        "model": Path("code/llm/language_model.py").read_text(encoding="utf-8"),
        "generation": Path("code/llm/generate.py").read_text(encoding="utf-8"),
    }
    assert "token_ids must have shape (batch_size, sequence_length)" in source_files["embedding"]
    assert "batch_size, num_tokens, _ = x.shape" in source_files["attention"]
    assert ").transpose(1, 2)" in source_files["attention"]
    assert "return self.vocabulary_projection(x)" in source_files["model"]
    assert "next_token_logits = all_logits[0, -1]" in source_files["generation"]


def test_pre_training_weight_optimisation_matches_training_step(client):
    response = client.get("/pre-training/weight-optimisation")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "<h1>Pre-training: weight optimisation</h1>" in html
    source_markup = (
        "For each iteration in our pre-training loop, we have a batch of inputs and targets. For each input/target pair, our model simultaneously calculates the probability of predicting the target token from the input tokens:",
        "We then use a <em>loss function</em> called <em>cross-entropy loss</em> to calculate how well our model did at predicting the correct tokens. This takes all of our predictions across all inputs in a given batch, and produces a single score called the <em>loss</em>.",
        "The reason for using <code>flatten</code> is that PyTorch cross-entropy expects a two-dimensional collection of predictions and one target ID for each prediction. Flattening does the following:",
        "Each flattened row of logits is still paired with exactly the same target token as before. Flattening changes only the layout expected by the loss function; it does not mix predictions and targets.",
        "Cross-entropy gives a larger loss when the correct token receives a low score and a smaller loss when it receives a high score. By default, PyTorch averages the loss across all <code>batch × tokens</code> predictions.",
        "Once the cross-entropy loss has been calculated, <em>backpropagation</em> calculates how sensitive that loss is to each weight in the model, i.e., how much changing that weight would likely change the loss.",
        "More precisely, it computes the partial derivative of the loss with respect to every weight. These derivatives collectively form the <em>gradient</em>.",
        "To do this, the model’s calculations are treated as a chain - or computational graph - of simple mathematical operations. Backpropagation works backward through these operations, repeatedly applying the differentiation chain rule to calculate how each weight may have contributed to the loss.",
        "Don’t worry if partial derivatives and the chain rule are unfamiliar; the key point is that backpropagation provides a systematic way to obtain the gradients needed to improve the model.",
        "Putting this all together, the code looks like the following:",
        "After each pass through the dataset, our script prints the average loss across its batches:",
        "This number is useful to show that optimisation is working, but for a real LLM a slightly more sophisticated training system is used, where some of our text corpus (called the <em>validation</em> <em>dataset</em>) is held back from training, and used to validate the model.",
    )
    for source_text in source_markup:
        assert source_text in html

    assert "<h2>Backpropagation</h2>" in html
    assert "<h2>Optimisation</h2>" in html
    assert "<h2>Monitor the training loss</h2>" in html
    assert 'class="ptd-table ptd-probabilities"' in html
    assert "The probability of “cat” is 0.004" in html
    assert "The probability of “sat” is 0.03" in html
    assert "The probability of “on” is 0.01" in html
    assert "The probability of “the” is 0.06" in html
    assert "The probability of “mat” is 0.02" in html
    assert 'id="cross-entropy-code"' in html
    assert "logits.flatten(0, 1)" in html
    assert "targets.flatten()" in html
    assert "F.cross_entropy" in html
    assert "Shapes passed to cross-entropy" in html
    assert "(batch × tokens, vocabulary)" in html
    assert 'id="chain-rule-title"' in html
    assert "∂L/∂w = ∂L/∂v × ∂v/∂u × ∂u/∂w = 10 × 1 × 4 = 40" in html
    assert "An optimiser then uses the resulting gradients" in html
    assert "The optimiser often used in LLMs" in html
    assert (
        'href="https://optimization.cbe.cornell.edu/index.php?title=AdamW">AdamW</a>'
        in html
    )
    assert 'id="optimisation-step-code"' in html
    assert "torch.optim.AdamW(model.parameters(), lr=learning_rate)" in html
    assert "optimiser.zero_grad()" in html
    assert "loss.backward()" in html
    assert "optimiser.step()" in html
    assert 'id="average-loss-code"' in html
    assert "Why no softmax?" not in html
    assert "The included script is intentionally direct" not in html
    assert 'href="/pre-training/full-loop"' in html

    source = Path("code/llm/pretrain.py").read_text(encoding="utf-8")
    for code_line in (
        "optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate)",
        "logits.flatten(0, 1)",
        "targets.flatten()",
        "optimiser.zero_grad()",
        "loss.backward()",
        "optimiser.step()",
    ):
        assert code_line in source


def test_pre_training_full_loop_matches_checkpoint_workflow(client):
    response = client.get("/pre-training/full-loop")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "<h1>Pre-training: the full training loop</h1>" in html
    assert (
        "We now have all the pieces needed to pre-train our model. The "
        in html
    )
    assert (
        'href="https://github.com/paolo2299-org/how-llms-work/tree/main/code/llm"'
        in html
    )
    assert "constructs shifted batches, initialises a model" in html
    assert 'id="pretrain-command-code"' in html
    assert "python code/llm/pretrain.py" in html
    assert 'id="model-and-data-code"' in html
    assert '"model_dim": 32' in html
    assert '"num_layers": 2' in html
    assert "batches = create_data_loader(" in html
    assert 'id="train-call-code"' in html
    assert 'id="save-checkpoint-code"' in html
    assert '"model_config": model_config' in html
    assert '"model_state": model.state_dict()' in html
    assert "In order to reconstruct the model later" in html
    assert 'id="generate-default-command-code"' in html
    assert "# implicitly loads weights/gpt2-small.pth" in html
    assert 'id="generate-checkpoint-command-code"' in html
    assert "--checkpoint weights/tiny-teaching-model.pth" in html
    assert 'id="single-prompt-batch-code"' in html
    assert "prompt_token_ids = tokenise(prompt)" in html
    assert "[prompt_token_ids]" in html
    assert "amount of training text dramatically" in html
    assert 'id="load-training-checkpoint-code"' not in html
    assert 'href="/pre-training/weight-optimisation"' in html
    assert 'href="/#post-training"' in html

    pretrain_source = Path("code/llm/pretrain.py").read_text(encoding="utf-8")
    generation_source = Path("code/llm/generate.py").read_text(encoding="utf-8")
    assert '"model_dim": 32' in pretrain_source
    assert '"num_layers": 2' in pretrain_source
    assert '"model_config": model_config' in pretrain_source
    assert '"model_state": model.state_dict()' in pretrain_source
    assert "prompt_token_ids = tokenise(prompt)" in generation_source
    assert "[prompt_token_ids]" in generation_source


def test_instruction_fine_tuning_deep_dive_matches_source_and_placeholders(client):
    response = client.get("/fine-tuning")
    html = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "<h1>Instruction fine-tuning</h1>" in html
    assert "a language model is good at text completion" in html
    assert "used as a chatbot" in html
    assert "if I were to ask" in html
    assert "What should I have for dinner tonight?" in html
    assert "Building the examples" in html
    assert "so-called <em>synthetic instruction tuning</em>" in html
    assert "synthetic instruction tuning" in html
    assert "examples might look like:</p>" in html
    assert html.count('class="ift-example"') == 3
    assert "Why does the Moon appear to change shape?" in html
    assert "Rewrite this politely" in html
    assert "Give me three names for a bakery" in html
    assert "<h2>Training</h2>" in html
    assert "Backpropagation and optimisation then make small adjustments" in html
    assert 'class="ift-loss-figure"' in html
    assert "A simplified response-only loss mask" in html
    assert html.count('class="ift-scored-token"') == 8
    assert "Context only" in html
    assert "Work in progress" not in html
    assert "&lt;insert possible" not in html
    assert "&lt;include the examples" not in html
    assert "&lt;visualisation" not in html
    assert 'href="/#post-training"' in html


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
