
import numpy as np
import torch
import torch.nn as nn


class CBOW(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, num_context_words, weigths=None, vector=None, train_weigths=False):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
        self.lin = nn.Linear(embedding_dim, num_embeddings, bias=False)

        assert isinstance(weigths, (torch.Tensor, np.ndarray, list, tuple)) or weigths is None
        
        try:
            weigths = torch.Tensor(weigths)
        except:
            if vector is False:
                weigths = torch.rand(num_context_words)
            elif vector is True:
                weigths = torch.rand(num_context_words, embedding_dim)
            else: # if vector is None:
                weigths = torch.ones(num_context_words)
            
        self.weigths = nn.Parameter(weigths, requires_grad=train_weigths)

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
