
import numpy as np
import torch
import torch.nn as nn


class CBOW(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, num_context_words=6, weights=None, vector=None, train_weights=False, shared_embedding=False):
        # By default, num_context_words=6 in order to be fully compatible with the original code from José Adrién Rodríguez Fonollosa
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
        self.lin = nn.Linear(embedding_dim, num_embeddings, bias=False)
        if shared_embedding : self.lin.weight.data = self.emb.weight.data.transpose(1, 0)

        assert isinstance(weights, (torch.Tensor, np.ndarray, list, tuple)) or weights is None
        
        try:
            weights = torch.Tensor(weights)
            assert len(weights) == num_context_words or (len(weights.shape) == 2 and weights.shape[0] == num_context_words and weights.shape[1] == embedding_dim)
            # docopts only consider handpicking 1D weights
        except: # if weights is None
            if vector is False:
                weights = torch.rand(num_context_words)
            elif vector is True:
                weights = torch.rand(num_context_words, embedding_dim)
            else: # if vector is None:
                weights = torch.ones(num_context_words)
            
        if len(weights.shape) == 1:
            weights = weights.unsqueeze(dim=0).unsqueeze(dim=2)
        elif len(weights.shape) == 2:
            weights = weights.unsqueeze(dim=0)
        else:
            assert len(weights.shape) == 1 or len(weights.shape) == 2

        self.weights = nn.Parameter(weights, requires_grad=train_weights)

    # B = Batch size
    # W = Number of context words (left + right)
    # E = embedding_dim
    # V = num_embeddings (number of words)
    def forward(self, input):
        # input shape is (B, W)
        e = self.emb(input)
        # e shape is (B, W, E)

        # u = e.sum(dim=1)
        # DONE: dot product through dim=1 to apply weights:
        u = (e * self.weights).sum(dim=1)
        # u shape is (B, E)

        z = self.lin(u)
        # z shape is (B, V)
        return z
