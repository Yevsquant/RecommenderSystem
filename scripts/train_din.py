"""
Train a Deep Interest Network (DIN) for ranking/CTR prediction.
"""
import torch
import numpy as np
from recommender.data import simulate_data
from recommender.ranker.din import DIN

def main():
    # Pseudo data
    SEED = 42
    torch.manual_seed(SEED)

    n_users, n_items, emb_dim, last_n = 300, 500, 32, 5
    interactions, meta = simulate_data(n_users=n_users, n_items=n_items, emb_dim=emb_dim)
    user_emb = torch.tensor(meta["user_emb"], dtype=torch.float32)
    item_emb = torch.tensor(meta["item_emb"], dtype=torch.float32)

    B = 256
    hist_emb = torch.randn(B, last_n, emb_dim)
    cand_emb = torch.randn(B, emb_dim)
    labels = torch.randint(0, 2, (B, 1)).float()

    model = DIN(emb_dim=emb_dim, last_n=last_n)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epo in range(5):
        optimizer.zero_grad()
        preds = model(hist_emb, cand_emb)
        loss = model.compute_loss(preds, labels)
        loss.backward()
        optimizer.step()
        print(f"[Epoch {epo+1}] Loss = {loss.item():.4f}")
    
    torch.save(model.state_dict(), "models/DIN.pt")
    print("✅ DIN model saved to models/DIN.pt")

if __name__ == "__main__":
    main()