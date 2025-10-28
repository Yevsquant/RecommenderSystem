import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiObjectiveRanker(nn.module):
    """
    Multi-objective ranking model
    Predict CTR, Like rate, Collect rate, and Share rate simultaneously
    """

    def __init__(self, input_dim: int, hidden_dims=(128,64), task_weights=None):
        super().__init__()
        # the layers shared by all four objectives
        self.shared_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
        )

        self.click_head = nn.Linear(hidden_dims[-1], 1)
        self.like_head = nn.Linear(hidden_dims[-1], 1)
        self.collect_head = nn.Linear(hidden_dims[-1], 1)
        self.share_head = nn.Linear(hidden_dims[-1], 1)

        self.task_weights = task_weights or {
            "click": 1.0, "like": 0.7, "collect": 0.5, "share": 0.3
        }

    def forward(self, x):
        h = self.shared_layers(x)
        return {
            "click": torch.sigmoid(self.click_head(h)),
            "like": torch.sigmoid(self.like_head(h)),
            "collect": torch.sigmoid(self.collect_head(h)),
            "share": torch.sigmoid(self.share_head(h))
        }
    
    def compute_loss(self, preds, targets):
        """Weighted multi-task binary cross-entropy loss"""
        total_loss = 0.0
        for t, weight in self.task_weights.items():
            loss = F.binary_cross_entropy(preds[t], targets[t])
            total_loss += weight * loss

        return total_loss