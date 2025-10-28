import numpy as np
import pandas as pd
from typing import List, Dict
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

class ItemCF:
    """
    Item-Based Collaborative Filtering (ItemCF)
    Uses co-occurrence or cosine similarity between item-user vectors.

    Customers who bought Item X also bought Item Y.
    => Similarity between X and Y

    Args:
        k_sim : Number of similar items to store per item.
    """
    def __init__(self, k_sim: int = 50):
        self.k_sim = k_sim
        self.item_similarity: Dict[int, Dict[int, float]] = {}
        self.user_item: Dict[int, set] = {}
        self.items: List[int] = []

    def fit(self, interactions: pd.DataFrame):
        """
        Build item-item similarity matrix based on user interactions.
        Args:
            interactions : DataFrame with cols ['user_id', 'item_id', 'clicked']
            'clicked' is a simple metric of the user2item rating
        """
        print("[ItemCF] Building item similarity matrix...")
        self.user_item = interactions.groupby("user_id")["item_id"].apply(set).to_dict()
        all_items = sorted(interactions["item_id"].unique())
        self.items = all_items
        item_index = {i: idx for idx, i in enumerate(all_items)}

        # Build item-user matrix
        n_items = len(all_items)
        n_users = interactions["user_id"].nunique()
        print(f"[ItemCF] {n_items} items, {n_users} users")
        item_user = interactions.groupby("item_id")["user_id"].apply(set).to_dict()

        # Compute cos similarity. Here we did co-occurrence similarity for simplicity
        item_item = np.zeros((n_items, n_items))
        for i, users_i in item_user.items():
            i_idx = item_index[i]
            for j, users_j in item_user.items():
                if i == j:
                    continue
                inter = len(users_i & users_j) # the overlap of users for item_i and item_j
                if inter == 0:
                    continue
                sim = inter / np.sqrt(len(users_i)*len(users_j))
                item_item[i_idx, item_index[j]] = sim
        
        # Keep top-k similar items for each item
        sim_df = pd.DataFrame(item_item, index=all_items, columns=all_items)
        self.item_similarity = {
            i: sim_df.loc[i].sort_values(ascending=False).head(self.k_sim).to_dict()
            for i in all_items
        }
        print("[ItemCF] Similarity matrix built.")

    def recommend(self, user_id: int, top_k: int = 10) -> List[int]:
        """
        Generate candidate items for a user.
        Still based on the occurrence not the user rating.
        """
        if self.item_similarity is None:
            raise ValueError("Model not fitted yet.")
        
        interacted_items = self.user_item.get(user_id, set())
        if not interacted_items:
            popular = sorted(
                self.items, key = lambda i: np.mean(list(self.item_similarity[i].values())),
                reverse=True
            )
            return popular[:top_k]
        
        scores = defaultdict(float)
        for i in interacted_items:
            for j, sim in self.item_similarity.get(i, {}).items():
                if j not in interacted_items:
                    scores[j] += sim
        ranked = sorted(scores.items(), key=lambda x: -x[1])

        return [i for i, _ in ranked[:top_k]]
