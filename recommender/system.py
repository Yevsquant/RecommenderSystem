# Combined RecommenderSystem Class
import pandas as pd
import torch
from .data import build_feature_row
from .candidate import candidate_generation
from .rerank import rerank_diversity
from recommender.ranker.multitask_ranker import MultiObjectiveRanker

class RecommenderSystem:
    def __init__(self, meta, ranker_model, item_cf: ItemCF):
        self.meta = meta
        self.ranker =ranker_model
    
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
        if isinstance(self.ranker, MultiObjectiveRanker):
            x = torch.tensor(feats[["dot", "cosine", "popularity", "age_days"]].values, dtype=torch.float32)

            with torch.no_grad():
                preds = self.ranker(x)
                scores = (
                    1.0 * preds["click"]
                    + 0.8 * preds["like"]
                    + 0.5 * preds["collect"]
                    + 0.3 * preds["share"]
                ).numpy().flatten()
                ranked = [i for _, i in sorted(zip(scores, candidates), reverse=True)]
        else:
            scores = self.ranker.predict_proba(feats)
            ranked = [candidates[iid] for iid in scores.argsort()[::-1][:top_k*2]]
        # rerank
        final = rerank_diversity(ranked[:top_k * 3], scores[:top_k * 3], self.meta)
        # final = rerank_diversity(ranked, scores[:len(ranked)], self.meta)[:top_k]
        return final