# Re-ranking Module
import numpy as np

def rerank_diversity(iids, scores, meta, lambda_div=0.5):
    """
    Maximum Marginal Relevance(MMR) via cos sim
    Take an initial list of recommended items and their prediction scores,
    Readjust the scores to PRIORITIZE diversity among the top recommendations.
    Advoid recommending identical items.
    """
    item_embs = meta["item_emb"]
    selected, scores_out = [], []
    # Highest scores to lowest scores
    for i, s in sorted(zip(iids, scores), key=lambda x: -x[1]):
        penalty = max([
            item_embs[i].dot(item_embs[j]) /
            (np.linalg.norm(item_embs[i])*np.linalg.norm(item_embs[j])+1e-9)
            for j in selected],
            default=0
        )
        selected.append(i)
        scores_out.append(s - lambda_div*penalty)
    order = np.argsort(-np.array(scores_out))
    return [selected[i] for i in order]