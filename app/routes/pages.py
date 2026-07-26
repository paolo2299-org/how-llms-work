from flask import Blueprint, render_template

pages_bp = Blueprint("pages", __name__)


@pages_bp.route("/")
def index():
    return render_template("index.html")


@pages_bp.route("/tokenisation")
def tokenisation():
    return render_template("tokenisation.html")


@pages_bp.route("/token-embeddings")
def token_embeddings():
    return render_template("token-embeddings.html")


@pages_bp.route("/self-attention")
def self_attention():
    return render_template("self-attention.html")


@pages_bp.route("/multi-head-attention")
def multi_head_attention():
    return render_template("multi-head-attention.html")


@pages_bp.route("/feed-forward")
def feed_forward():
    return render_template("feed-forward.html")


@pages_bp.route("/transformer-block")
def transformer_block():
    return render_template("transformer-block.html")


@pages_bp.route("/full-llm")
def full_llm():
    return render_template("full-llm.html")


@pages_bp.route("/open-weights")
def open_weights():
    return render_template(
        "work-in-progress.html",
        page_title="Open Weights",
        section_id="open-weights",
    )


@pages_bp.route("/pre-training")
def pre_training():
    return render_template(
        "work-in-progress.html",
        page_title="Pre-training",
        section_id="pre-training",
    )


@pages_bp.route("/fine-tuning")
def fine_tuning():
    return render_template(
        "work-in-progress.html",
        page_title="Fine-tuning",
        section_id="fine-tuning",
    )
