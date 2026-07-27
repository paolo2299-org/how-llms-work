"""A language model that supports batched inference and pre-training."""

import torch.nn as nn

from token_embedding import TokenAndPositionEmbedding
from transformer import TransformerBlock


class LanguageModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        max_sequence_length,
        model_dim,
        head_dim,
        num_heads,
        hidden_dim,
        num_layers,
        qkv_bias=False,
    ):
        super().__init__()
        self.embedding = TokenAndPositionEmbedding(
            vocab_size,
            max_sequence_length,
            model_dim,
        )
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    model_dim,
                    head_dim,
                    num_heads,
                    hidden_dim,
                    qkv_bias=qkv_bias,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(model_dim)
        self.vocabulary_projection = nn.Linear(
            model_dim,
            vocab_size,
            bias=False,
        )

    def forward(self, token_ids):
        if token_ids.ndim != 2:
            raise ValueError(
                "token_ids must have shape (batch_size, sequence_length)"
            )
        if token_ids.size(1) == 0:
            raise ValueError("each sequence must contain at least one token")

        x = self.embedding(token_ids)

        for block in self.transformer_blocks:
            x = block(x)

        x = self.final_norm(x)

        # Training needs a next-token prediction at every input position.
        return self.vocabulary_projection(x)
