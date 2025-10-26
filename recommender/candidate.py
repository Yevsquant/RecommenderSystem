# Candidate Generator
import numpy as np

def candidate_generation(uid, meta, top_n=200):
    uemb = meta["user_emb"][uid]
    item_embs = meta["item_emb"]
    scores = item_embs @ uemb
    top_idx = np.argsort(-scores)[:top_n]
    return top_idx.tolist()