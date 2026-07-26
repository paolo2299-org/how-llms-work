import torch.nn as nn

from feed_forward import FeedForwardLayer
from self_attention import MultiHeadAttention


class TransformerBlock(nn.Module):

    def __init__(self, model_dim, head_dim, num_heads, hidden_dim):
        super().__init__()
        self.attention_norm = nn.LayerNorm(model_dim)
        self.attention = MultiHeadAttention(model_dim, head_dim, num_heads)
        self.feed_forward_norm = nn.LayerNorm(model_dim)
        self.feed_forward = FeedForwardLayer(model_dim, hidden_dim)

    def forward(self, x):
        x = x + self.attention(self.attention_norm(x))
        x = x + self.feed_forward(self.feed_forward_norm(x))
        return x
