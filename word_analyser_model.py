
import numpy as np


def cosine_similarity(x1, x2):
    return np.dot(x1, x2) / (np.norm(x1) * np.norm(x2))

class WordVectors:
    def __init__(self, vectors, vocabulary):
        # TODO 
        self.vocabulary = vocabulary
    
    def most_similar(self, word, topn=10):
        # TODO
        return [
            ('valencià', 0.8400525),
            ('basc', 0.75919044),
            ('gallec', 0.7418786),
            ('mallorquí', 0.73923385),
            ('castellà', 0.69002914),
            ('francès', 0.6782388),
            ('espanyol', 0.6611247),
            ('bretó', 0.641976),
            ('aragonès', 0.6250948),
            ('andalús', 0.6203275)
        ]
    
    def analogy(self, x1, x2, y1, topn=5, keep_all=False):
        # If keep_all is False we remove the input words (x1, x2, y1) from the returned closed words
        # TODO
        return [
            ('polonès', 0.9679756),
            ('suec', 0.9589857),
            ('neerlandès', 0.95811903),
            ('rus', 0.95155054),
            ('txec', 0.950968),
            ('basc', 0.94935954),
            ('danès', 0.94827694),
            ('turc', 0.9475782)
        ]

