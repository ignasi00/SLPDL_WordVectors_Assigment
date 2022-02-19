
import numpy as np
import torch
import torch.nn as nn


class CBOW(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, num_context_words, weigths=None):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
        self.lin = nn.Linear(embedding_dim, num_embeddings, bias=False)

        if isinstance(weigths, (torch.Tensor, np.ndarray, list, tuple)):
            self.weigths = nn.Parameter(torch.Tensor(weigths), requires_grad=False)
            assert len(self.weigths.shape) == 1
        elif isinstance(weigths, torch.dtype):
            self.weigths = nn.Parameter(torch.rand(num_context_words, dtype=weigths), requires_grad=True)
        else: # Includes it being None (default)
            self.weigths = nn.Parameter(torch.ones(num_context_words), requires_grad=False)

    # B = Batch size
    # W = Number of context words (left + right)
    # E = embedding_dim
    # V = num_embeddings (number of words)
    def forward(self, input):
        # input shape is (B, W)
        e = self.emb(input)
        # e shape is (B, W, E)

        # u = e.sum(dim=1)
        # DONE: dot product through dim=1 to apply weigths:
        u = (e * self.weigths.unsqueeze(dim=0).unsqueeze(dim=2)).sum(dim=1)
        # u shape is (B, E)

        z = self.lin(u)
        # z shape is (B, V)
        return z
