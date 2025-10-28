# Combined RecommenderSystem Class
import pandas as pd
import torch
from .data import build_feature_row
from .candidate import candidate_generation
from .rerank import rerank_diversity
from recommender.retrieval.item_cf import ItemCF
from recommender.ranker.factorization_machine import FactorizationMachine

class RecommenderSystem:
    def __init__(self, meta, ranker_model=None, item_cf=None):
        self.meta = meta
        self.ranker =ranker_model
        self.item_cf = item_cf
    
    def recommend(self, uid, top_k=10):
        # retrieval
        if self.item_cf:
            candidates = self.item_cf.recommend(uid, top_k=200)
        else:
            candidates = candidate_generation(uid, self.meta, top_n=200)
        feats = pd.DataFrame([build_feature_row(uid, iid, self.meta) for iid in candidates])
        # ranker
        scores = None
        ranked = None
        if isinstance(self.ranker, FactorizationMachine):
            x = torch.tensor(feats[["dot", "cosine", "popularity", "age_days"]].values, dtype=torch.float32)

            with torch.no_grad():
                preds = self.ranker(x)
                scores = self.ranker(x).numpy().flatten()
                ranked = [i for _, i in sorted(zip(scores, candidates), reverse=True)]
        else:
            scores = self.ranker.predict_proba(feats)
            ranked = [candidates[iid] for iid in scores.argsort()[::-1][:top_k*2]]
        # rerank
        final = rerank_diversity(ranked[:top_k * 3], scores[:top_k * 3], self.meta)
        # final = rerank_diversity(ranked, scores[:len(ranked)], self.meta)[:top_k]
        return final