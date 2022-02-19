
import torch
import torch.nn as nn


class CBOW(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, num_context_words, weigths=None):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings, embedding_dim, padding_idx=0)
        self.lin = nn.Linear(embedding_dim, num_embeddings, bias=False)

        # TODO: if None => ; if vector => not trainable; if dtype => trainable
        if isinstance(weigths, (list, torch.Tensor)): # TODO: cualquier cosa que sirva para meter numeros a pelo
            self.weigths = # TODO: requier_grad=False Tensor of the elements of the weigths vector.
        elif : # TODO: comprobar si weigths es un dtype (int, float, etc)
            self.weigths = # TODO: trainable Tensor of num_context_words
        else: # Includes it being None
            self.weigths = # TODO: requier_grad=False Tensor of ones (num_context_words elements)


    # B = Batch size
    # W = Number of context words (left + right)
    # E = embedding_dim
    # V = num_embeddings (number of words)
    def forward(self, input):
        # input shape is (B, W)
        e = self.emb(input)
        # e shape is (B, W, E)
        # u = e.sum(dim=1) # TODO: dot product through dim=1 to apply weigths.
        u = (e * self.weigths.unsqueeze(dim=1)).sum(dim=1, keepdims=True)
        # u shape is (B, E)
        z = self.lin(u)
        # z shape is (B, V)
        return z
