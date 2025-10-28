import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from recommender.ranker.multitask_ranker import MultiObjectiveRanker

# batch_size, n_features = 32, 20
# x = torch.randn(batch_size, n_features)
# y = {
#     "click": torch.randint(0, 2, (batch_size, 1)).float(),
#     "like": torch.randint(0, 2, (batch_size, 1)).float(),
#     "collect": torch.randint(0, 2, (batch_size, 1)).float(),
#     "share": torch.randint(0, 2, (batch_size, 1)).float(),
# }

# model = MultiObjectiveRanker(input_dim=n_features)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# for epoch in range(5):
#     optimizer.zero_grad()
#     preds = model(x)
#     loss = model.compute_loss(preds, y)
#     loss.backward()
#     optimizer.step()
#     print(f"Epoch {epoch+1}, loss = {loss.item():.4f}")

class RankingDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feature_cols: list, label_cols: list):
        self.X = torch.tensor(df[feature_cols].values, dtype=torch.float32)
        self.Y = {label: torch.tensor(df[label].values, dtype=torch.float32).unsqueeze(1)
                  for label in label_cols}
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], {k: v[idx] for k, v in self.Y.items()}
        
def train_ranker(df, feature_cols, label_cols, input_dim, epochs=5, batch_size=64, lr=1e-3):
    dataset = RankingDataset(df, feature_cols, label_cols)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = MultiObjectiveRanker(input_dim=input_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0
        for x, y in loader:
            optimizer.zero_grad()
            preds = model(x)
            loss = model.compute_loss(preds, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"[Epoch {epoch+1}] Loss: {total_loss/len(loader):.4f}")

    return model

def evaluate(model, df, feature_cols, label_cols):
    dataset = RankingDataset(df, feature_cols, label_cols)
    loader = DataLoader(dataset, batch_size=128, shuffle=False)
    with torch.no_grad():
        losses = []
        for x, y in loader:
            preds = model(x)
            loss = model.compute_loss(preds, y)
            losses.append(loss.item())
    return sum(losses) / len(losses)

        