# Ranking Model
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

class BaseRanker:
    def __init__(self):
        self.model = LogisticRegression(max_iter=200)
        self.scaler  = StandardScaler() # Normalize features
        self.feature_cols = ["dot", "cosine", "popularity", "age_days"]

    def fit(self, df):
        X = self.scaler.fit_transform(df[self.feature_cols])
        y = df["clicked"]
        self.model.fit(X, y)

    def predict_proba(self, df):
        X = self.scaler.transform(df[self.feature_cols])
        return self.model.predict_proba(X)[:, 1]