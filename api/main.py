# FastAPI server exposing recommendations
from fastapi import FastAPI, Query
from pydantic import BaseModel
import random
import os
import torch

from recommender.data import simulate_data
from recommender.retrieval.item_cf import ItemCF
from recommender.retrieval.base_retrieval import BaseRetrieval
from recommender.ranker.baseranker import BaseRanker
from recommender.ranker.multitask_ranker import MultiObjectiveRanker
from recommender.ranker.factorization_machine import FactorizationMachine
from recommender.ranker.din import DIN
from recommender.system import RecommenderSystem

RETRIEVAL_CLASS_MAP = {
    "ItemCF": ItemCF,
    "BaseRetrieval": BaseRetrieval,
}

RANKER_CLASS_MAP = {
    "MultiObjectiveRanker": MultiObjectiveRanker,
    "FactorizationMachine": FactorizationMachine,
    "DIN": DIN,
    "BaseRanker": BaseRanker
}

retrieval_strategy_name = os.environ.get("RETRIVAL", "BaseRetrieval")
ranker_strategy_name = os.environ.get("RANKER", "BaseRanker")

retrieval = RETRIEVAL_CLASS_MAP[retrieval_strategy_name]()

app = FastAPI(title="Recommender System API")

# Get data
interactions, meta = simulate_data()

# Set up retrieval
if isinstance(retrieval, BaseRetrieval):
    retrieval.fit(meta)
else:
    retrieval.fit(interactions)
print(f"Retrieval: {retrieval_strategy_name} ...")

# Set up ranker
if ranker_strategy_name == "BaseRanker":
    ranker = BaseRanker()
    ranker.fit(interactions.assign(dot=0, cosine=0, popularity=5, age_days=100))
else:
    if ranker_strategy_name == "MultiObjectiveRanker":
        ranker = MultiObjectiveRanker(input_dim=4) # same with x in system.py
    if ranker_strategy_name == "FactorizationMachine":
        ranker = FactorizationMachine(n_features=4) # same with x in system.py
    if ranker_strategy_name == "DIN":
        ranker = DIN()

    print(f"Ranker: {ranker_strategy_name} ...")
    model_path = f"models/{ranker_strategy_name}.pt"
    try:
        state_dict = torch.load(model_path)
    except Exception as e:
        print(f"Error loading model file: {e}")
    ranker.load_state_dict(state_dict)
    ranker.eval()
    
    

# Set up recommender system
recsys = RecommenderSystem(meta, interactions, retrieval, ranker)

class RecResponse(BaseModel):
    user_id: int
    recommendations: list[int]

@app.get("/recommend", response_model=RecResponse)
def get_recommendations(user_id: int = Query(0, ge=0)):
    recs = recsys.recommend(user_id, top_k=10)
    return {"user_id": user_id, "recommendations": recs}