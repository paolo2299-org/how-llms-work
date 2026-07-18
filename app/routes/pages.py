from flask import Blueprint, render_template

pages_bp = Blueprint("pages", __name__)


@pages_bp.route("/")
def index():
    return render_template("index.html")


@pages_bp.route("/self-attention")
def self_attention():
    return render_template("self-attention.html")


@pages_bp.route("/multi-head-attention")
def multi_head_attention():
    return render_template("multi-head-attention.html")


@pages_bp.route("/feed-forward")
def feed_forward():
    return render_template("feed-forward.html")
