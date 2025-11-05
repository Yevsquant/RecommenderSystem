import torch
import torch.nn as nn
import torch.nn.functional as F

class FactorizationMachine(nn.Module):
    """
    Second-order Factorization Machine (FM)
    Reference: Steffen Rendle, "Factorization Machines", ICDM 2010
    """

    def __init__(self, n_features: int, k_factors: int = 16):
        super().__init__()
        self.n_features = n_features
        self.k_factors = k_factors

        self.bias = nn.Parameter(torch.zeros(1))
        self.weights = nn.Parameter(torch.randn(n_features))
        self.v = nn.Parameter(torch.randn(n_features, k_factors))

        nn.init.xavier_uniform_(self.v)

    def forward(self, x):
        linear = torch.matmul(x, self.weights) + self.bias

        # .5 * ((xv)^2 - (x^2)(v^2))
        xv = torch.matmul(x, self.v)
        xv_sq = xv * xv
        x_sq_v_sq = torch.matmul(x*x, self.v*self.v)
        interaction = .5 * torch.sum(xv_sq - x_sq_v_sq)

        y = linear.unsqueeze(1) + interaction
        return torch.sigmoid(y)
    
    def compute_loss(self, preds, targets):
        return F.binary_cross_entropy(preds, targets)