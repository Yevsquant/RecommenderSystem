from recommender.retrieval.item_cf import ItemCF
from recommender.data import simulate_data

def test_item_cf_recommend():
    interactions, _ = simulate_data(n_users=50, n_items=30, n_interactions=200)
    model = ItemCF(k_sim=10)
    model.fit(interactions)
    recs = model.recommend(0, top_k=5)
    assert isinstance(recs, list) and len(recs) > 0