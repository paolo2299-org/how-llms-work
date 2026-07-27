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
        if token_ids.ndim != 1:
            raise ValueError("token_ids must be a single sequence")
        if token_ids.numel() == 0:
            raise ValueError("token_ids must contain at least one token")

        x = self.embedding(token_ids)

        for block in self.transformer_blocks:
            x = block(x)

        x = self.final_norm(x)
        # Training needs a prediction at every position: each token is used to
        # predict the token that follows it. Inference simply selects the last
        # row of these logits.
        return self.vocabulary_projection(x)
