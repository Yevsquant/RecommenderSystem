import torch
import torch.nn as nn
import torch.nn.functional as F

class DIN(nn.Module):
    """
    Deep Interest Network (DIN)
    Reference: Zhou et al., KDD 2018
    Uses attention between candidate item and user's LastN items.
    """

    def __init__(self, emb_dim: int = 32, hidden_dims=(128, 64), last_n: int = 5):
        super().__init__()
        self.emb_dim = emb_dim
        self.last_n = last_n

        self.attn = nn.Sequential(
            nn.Linear(4 * emb_dim, 80),
            nn.ReLU(),
            nn.Linear(80, 40),
            nn.ReLU(),
            nn.Linear(40, 1)
        )

        self.mlp = nn.Sequential(
            nn.Linear(2 * emb_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1),
            nn.Sigmoid()
        )
    
    def forward(self, hist_emb: torch.Tensor, cand_emb: torch.Tensor):
        """
        Args:
            hist_emb: (batch, N, D)  - embeddings of user's LastN items
            cand_emb: (batch, D)     - candidate item embeddings
        """
        B, N, D = hist_emb.shape
        q = cand_emb.unsqueeze(1).expand(-1, N, -1)

        # Build input for attention MLP: [q, h, q - h, q * h]
        att_inp = torch.cat([q, hist_emb, q - hist_emb, q * hist_emb], dim=-1)
        att_w = self.attn(att_inp).squeeze(-1) # (B,N)
        att_w = F.softmax(att_w, dim=1)

        user_interest = torch.sum(att_w.unsqueeze(-1) * hist_emb, dim=1) # (B,D)

        x = torch.cat([user_interest, cand_emb], dim=-1)
        out = self.mlp(x)
        return out
    
    def compute_loss(self, preds, targets):
        return F.binary_cross_entropy(preds, targets)