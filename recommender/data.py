# Data Simulation & Feature Engineering
import numpy as np
import pandas as pd

def simulate_data(n_users=500, n_items=800, n_interactions=8000, emb_dim=16, seed=42):
    rng = np.random.default_rng(seed) # Random Number Generator
    user_emb = rng.normal(size=(n_users, emb_dim)) # Representation of users (by vectors)
    item_emb = rng.normal(size=(n_items, emb_dim)) # arr shape(n, emb_dim)
    items = pd.DataFrame({
        "item_id": np.arange(n_items),
        "popularity": rng.poisson(5, size=n_items),
        "age_days": rng.integers(0, 365, size=n_items),
    })
    interactions = pd.DataFrame({
        "user_id": rng.integers(0, n_users, size=n_interactions),
        "item_id": rng.integers(0, n_items, size=n_interactions),
    })
    interactions["clicked"] = (rng.random(n_interactions) < 0.2).astype(int) # Click Through Rate (CTR)
    meta = {"user_emb": user_emb, "item_emb": item_emb, "item_df": items}
    return interactions, meta

def build_feature_row(uid, iid, meta):
    """Calculate features of a user-item pair"""
    uemb, iemb = meta["user_emb"][uid], meta["item_emb"][iid]
    return {
        "user_id": uid, "item_id": iid,
        "dot": float(uemb.dot(iemb)),
        "cosine": float(uemb.dot(iemb)/(np.linalg.norm(uemb)*np.linalg.norm(iemb)+1e-9)),
        "popularity": float(meta["item_df"].loc[iid, "popularity"]),
        "age_days": float(meta["item_df"].loc[iid, "age_days"]),
    }

def featurize(interactions: pd.DataFrame, meta: dict) -> pd.DataFrame:
    """
    Build a feature dataframe for ranking model training
    Args:
        interactions : DataFrame
            with cols: user_id, item_id, clicked (and other engagement labels if available)
        meta : dict
            contains embeddings and item metadata:
            { "user_emb": np.ndarray, "item_emb": np.ndarray, "items_df": pd.DataFrame}
    Returns:
        pd.DataFrame
            Feature dataframe with columns: [user_id, item_id, dot, cosine, popularity, age_days]
    """
    user_emb = meta["user_emb"]
    item_emb = meta["item_emb"]
    items_df = meta["item_df"]

    feats = []
    for _, row in interactions.iterrows():
        u = int(row.user_id)
        i = int(row.item_id)

        feat = build_feature_row(u, i, meta)
        # Preserve existing label columns (clicked, liked, etc.)
        for col in interactions.columns:
            if col not in feat and col not in ["user_id", "item_id"]:
                feat[col] = row[col]
        feats.append(feat)

    return pd.DataFrame(feats)