import torch
import numpy as np
from recommender.data import simulate_data, featurize
from recommender.ranker.factorization_machine import FactorizationMachine

# Train a Factorization Machine (FM) for ranking (CTR prediction).
def main():
    # 1. Generate training data
    interactions, meta = simulate_data(n_users=400, n_items=500, n_interactions=5000)
    df = featurize(interactions, meta)

    feature_cols = ["dot", "cosine", "popularity", "age_days"]

    # 2. Build tensors
    X = torch.tensor(df[feature_cols].values, dtype=torch.float32)
    y = torch.tensor(df["clicked"].values, dtype=torch.float32).unsqueeze(1)

    # 3. Initialize FM
    model = FactorizationMachine(n_features=len(feature_cols), k_factors=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # 4. Train
    for epoch in range(5):
        optimizer.zero_grad()
        preds = model(X)
        loss = model.compute_loss(preds, y)
        loss.backward()
        optimizer.step()
        print(f"[Epoch {epoch+1}] Loss = {loss.item():.4f}")
    
    # 5. Save model
    torch.save(model.state_dict(), "models/fm_ranker.pt")
    print("✅ FM model saved to models/fm_ranker.pt")


if __name__ == "__main__":
    main()
