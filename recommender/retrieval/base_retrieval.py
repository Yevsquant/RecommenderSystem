# Candidate Generator
import numpy as np
class BaseRetrieval():
    def __init__(self):
        self.meta = None
    
    def fit(self, meta):
        self.meta = meta

    def candidate_generation(self, user_id, top_n=200):
        uemb = self.meta["user_emb"][user_id]
        item_embs = self.meta["item_emb"]
        scores = item_embs @ uemb
        top_idx = np.argsort(-scores)[:top_n]
        return top_idx.tolist()