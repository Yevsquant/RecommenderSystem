# FastAPI server exposing recommendations
from fastapi import FastAPI, Query
from pydantic import BaseModel
import random
from recommender.data import simulate_data
from recommender.ranker import Ranker
from recommender.system import RecommenderSystem

app = FastAPI(title="Recommender System API")

interactions, meta = simulate_data()
ranker = Ranker()
ranker.fit(interactions.assign(dot=0, cosine=0, popularity=5, age_days=100))
recsys = RecommenderSystem(meta, ranker)

class RecResponse(BaseModel):
    user_id: int
    recommendations: list[int]

@app.get("/recommend", response_model=RecResponse)
def get_recommendations(user_id: int = Query(0, ge=0)):
    recs = recsys.recommend(user_id, top_k=10)
    return {"user_id": user_id, "recommendations": recs}