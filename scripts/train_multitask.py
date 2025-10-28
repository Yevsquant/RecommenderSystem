import torch
import numpy as np
from recommender.data import simulate_data, featurize
from recommender.ranker.trainer import train_ranker
from recommender.system import RecommenderSystem

def main():
    # 1. Generate training data
    interactions, meta = simulate_data(n_users=400, n_items=500, n_interactions=5000)
    df = featurize(interactions, meta)

    # Add fake labels for demo
    for col in ["click", "like", "collect", "share"]:
        df[col] = (df["dot"] + np.random.randn(len(df))) > 0

    feature_cols = ["dot", "cosine", "popularity", "age_days"]
    label_cols = ["click", "like", "collect", "share"]

    # 2. Train model
    ranker = train_ranker(df, feature_cols, label_cols, input_dim=len(feature_cols), epochs=5)
    torch.save(ranker.state_dict(), "models/multitask_ranker.pt")

    # 3. Integrate into full recommender
    recsys = RecommenderSystem(meta, ranker_model=ranker)

    # 4. Serve recommendations
    recs = recsys.recommend(uid=0, top_k=10)
    print("Top-10 recommendations:", recs)

if __name__ == "__main__":
    main()
