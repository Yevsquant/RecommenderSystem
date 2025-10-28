# Combined RecommenderSystem Class
import pandas as pd
from .data import build_feature_row
from .candidate import candidate_generation
from .rerank import rerank_diversity

class RecommenderSystem:
    def __init__(self, meta, ranker):
        self.meta = meta
        self.ranker =ranker
    
    def recommend(self, uid, top_k=10):
        # retrieval
        candidates = candidate_generation(uid, self.meta, top_n=200)
        feats = pd.DataFrame([build_feature_row(uid, iid, self.meta) for iid in candidates])
        # ranker
        scores = self.ranker.predict_proba(feats)
        ranked = [candidates[iid] for iid in scores.argsort()[::-1][:top_k*2]]
        # rerank
        return rerank_diversity(ranked, scores[:len(ranked)], self.meta)[:top_k]