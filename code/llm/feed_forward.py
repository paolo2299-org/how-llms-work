"""The feed-forward part of the trainable transformer."""

import torch.nn as nn


class FeedForwardLayer(nn.Module):

    def __init__(self, model_dim, hidden_dim):
        super().__init__()
        self.expand = nn.Linear(model_dim, hidden_dim)
        self.activation = nn.GELU(approximate="tanh")
        self.project = nn.Linear(hidden_dim, model_dim)

    def forward(self, x):
        hidden = self.expand(x)
        activated = self.activation(hidden)
        return self.project(activated)
