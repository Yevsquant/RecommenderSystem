# Combined RecommenderSystem Class
import pandas as pd
import numpy as np
import torch
from typing import List
from .data import build_feature_row
from .rerank import rerank_diversity
from recommender.ranker.factorization_machine import FactorizationMachine
from recommender.ranker.multitask_ranker import MultiObjectiveRanker
from recommender.ranker.din import DIN

class RecommenderSystem:
    def __init__(self, meta, interactions, retrieval, ranker):
        self.meta = meta
        self.retrieval = retrieval
        self.ranker =ranker
        self.interactions = interactions

    def get_last_n_items(self, user_id: int, N: int = 5) -> List[int]:
        """Fetch last N interacted items (simulate with random sampling)."""
        user_hist = self.interactions.groupby("user_id")["item_id"].apply(list).to_dict().get(user_id)
        if user_hist is None or len(user_hist) == 0: # No relevant history => randomly pick items
            n_items = self.meta["item_emb"].shape[0]
            return list(np.random.choice(range(n_items), size=N, replace=False))
        elif len(user_hist) >= N:
            return user_hist[-N:]
        else:
            selected_items = user_hist[:]
            selected_items.extend(list(np.random.choice(range(n_items), size=N-len(user_hist), replace=False)))
            return selected_items

    
    def recommend(self, uid, top_k=10):
        # Retrieval: candidates is a list of item ids
        candidates = self.retrieval.candidate_generation(user_id=uid, top_n=200)
        # feats = (top_n,4)
        feats = pd.DataFrame([build_feature_row(uid, iid, self.meta) for iid in candidates])
    
        # ranker
        scores = None
        ranked = None
        if isinstance(self.ranker, FactorizationMachine):
            x = torch.tensor(feats[["dot", "cosine", "popularity", "age_days"]].values, dtype=torch.float32)
            with torch.no_grad():
                preds = self.ranker(x)
                scores = preds.numpy().flatten()

        elif isinstance(self.ranker, MultiObjectiveRanker):
            x = torch.tensor(feats[["dot", "cosine", "popularity", "age_days"]].values, dtype=torch.float32)
            with torch.no_grad():
                preds = self.ranker(x)
                scores = (
                    1.0 * preds["click"]
                    + 0.8 * preds["like"]
                    + 0.5 * preds["collect"]
                    + 0.3 * preds["share"]
                ).numpy().flatten()
    
        elif isinstance(self.ranker, DIN):
            last_items = self.get_last_n_items(uid, N=self.ranker.last_n)
            hist_emb = torch.tensor(self.meta["item_emb"][last_items], dtype=torch.float32).unsqueeze(0) #(1,N,D)
            cand_embs = torch.tensor(self.meta["item_emb"][candidates], dtype=torch.float32) #(top_n,D)
            hist_emb = hist_emb.repeat(len(candidates), 1, 1) #(top_n,N,D)
            with torch.no_grad():
                scores = self.ranker(hist_emb, cand_embs).numpy().flatten()
    
        else:
            scores = self.ranker.predict_proba(feats)

        # ranked = [candidates[iid] for iid in scores.argsort()[::-1][:top_k]]: faster
        ranked = [i for _, i in sorted(zip(scores, candidates), reverse=True)]

        # rerank
        # final = rerank_diversity(ranked[:top_k * 3], scores[:top_k * 3], self.meta): a little bit larger than top_k for rerank
        final = rerank_diversity(ranked, scores[:len(ranked)], self.meta)[:top_k]
        return final