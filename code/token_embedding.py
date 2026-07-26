import torch
import torch.nn as nn


class TokenAndPositionEmbedding(nn.Module):

    def __init__(self, vocab_size, max_sequence_length, model_dim):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.token_embedding = nn.Embedding(vocab_size, model_dim)
        self.position_embedding = nn.Embedding(max_sequence_length, model_dim)

    def forward(self, token_ids):
        if token_ids.ndim != 1:
            raise ValueError("token_ids must be a single sequence")
        if token_ids.size(0) > self.max_sequence_length:
            raise ValueError(
                f"prompt has {token_ids.size(0)} tokens, but the maximum is "
                f"{self.max_sequence_length}"
            )

        positions = torch.arange(
            token_ids.size(0),
            device=token_ids.device,
        )
        token_vectors = self.token_embedding(token_ids)
        position_vectors = self.position_embedding(positions)
        return token_vectors + position_vectors
