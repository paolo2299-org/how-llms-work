from pathlib import Path

import torch

from language_model import LanguageModel


DEFAULT_WEIGHTS_PATH = (
    Path(__file__).resolve().parents[2] / "weights" / "gpt2-small.pth"
)

TOP_LEVEL_NAMES = {
    "tok_emb.weight": "embedding.token_embedding.weight",
    "pos_emb.weight": "embedding.position_embedding.weight",
    "final_norm.scale": "final_norm.weight",
    "final_norm.shift": "final_norm.bias",
    "out_head.weight": "vocabulary_projection.weight",
}

BLOCK_NAMES = {
    "att.W_query.weight": "attention.W_query.weight",
    "att.W_query.bias": "attention.W_query.bias",
    "att.W_key.weight": "attention.W_key.weight",
    "att.W_key.bias": "attention.W_key.bias",
    "att.W_value.weight": "attention.W_value.weight",
    "att.W_value.bias": "attention.W_value.bias",
    "att.out_proj.weight": "attention.out_proj.weight",
    "att.out_proj.bias": "attention.out_proj.bias",
    "ff.layers.0.weight": "feed_forward.expand.weight",
    "ff.layers.0.bias": "feed_forward.expand.bias",
    "ff.layers.2.weight": "feed_forward.project.weight",
    "ff.layers.2.bias": "feed_forward.project.bias",
    "norm1.scale": "attention_norm.weight",
    "norm1.shift": "attention_norm.bias",
    "norm2.scale": "feed_forward_norm.weight",
    "norm2.shift": "feed_forward_norm.bias",
}


def translate_name(source_name):
    if source_name in TOP_LEVEL_NAMES:
        return TOP_LEVEL_NAMES[source_name]

    for block_number in range(12):
        prefix = f"trf_blocks.{block_number}."
        if source_name.startswith(prefix):
            block_name = source_name.removeprefix(prefix)
            try:
                translated_name = BLOCK_NAMES[block_name]
            except KeyError as error:
                raise ValueError(
                    f"unknown checkpoint parameter: {source_name}"
                ) from error
            return f"transformer_blocks.{block_number}.{translated_name}"

    raise ValueError(f"unknown checkpoint parameter: {source_name}")


def translate_weights(source_state, model_state):
    translated_state = {}

    for source_name, tensor in source_state.items():
        if source_name.endswith(".att.mask"):
            continue

        destination_name = translate_name(source_name)
        if destination_name in translated_state:
            raise ValueError(
                f"more than one checkpoint parameter maps to {destination_name}"
            )
        translated_state[destination_name] = tensor

    missing_names = model_state.keys() - translated_state.keys()
    unexpected_names = translated_state.keys() - model_state.keys()
    if missing_names or unexpected_names:
        raise ValueError(
            "checkpoint parameters do not match the model: "
            f"missing={sorted(missing_names)}, "
            f"unexpected={sorted(unexpected_names)}"
        )

    for name, tensor in translated_state.items():
        if tensor.shape != model_state[name].shape:
            raise ValueError(
                f"{name} has shape {tuple(tensor.shape)}, "
                f"expected {tuple(model_state[name].shape)}"
            )

    return translated_state


def load_gpt2_small(weights_path=DEFAULT_WEIGHTS_PATH, device="cpu"):
    model = LanguageModel(
        vocab_size=50257,
        max_sequence_length=1024,
        model_dim=768,
        head_dim=64,
        num_heads=12,
        hidden_dim=3072,
        num_layers=12,
        qkv_bias=True,
    )

    source_state = torch.load(
        Path(weights_path).expanduser(),
        map_location="cpu",
        weights_only=True,
    )
    translated_state = translate_weights(source_state, model.state_dict())

    model.load_state_dict(translated_state, strict=True)
    model.eval()
    return model.to(device)
