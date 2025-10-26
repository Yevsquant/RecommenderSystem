from recommender.data import simulate_data
from recommender.ranker import Ranker
from recommender.system import RecommenderSystem

def test_recommender_pipeline():
    interactions, meta = simulate_data()
    ranker = Ranker()
    ranker.fit(interactions.assign(dot=0, cosine=0, popularity=5, age_days=100))
    recsys = RecommenderSystem(meta, ranker)
    recs = recsys.recommend(0, top_k=5)
    assert isinstance(recs, list) and len(recs) == 5