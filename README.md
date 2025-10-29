# Recommender System

### Run the api:
`
uv run unicorn api.main:app --reload
`

### Run the test:
`
uv run pytest tests/<testfile.py>
`

### Open:
`
http://localhost:8000/docs
`

## Todo:
- [ ] ***Workflow Chaining***: Make the Recommand System run with arbitrary implemented techniques.
- [ ] ***retrieval/item_cf.py***: Require a more complex metric to compute similarity of items. Currently the similarity is computed via Co-Occurrence.
- [ ] ***retrieval/item_cf.py***: Implement a index database for user2item like matrix, item2item similarity matrix, and user2item recommend matrix.
- [ ] ***ranker/trainer.py***: Replace random input x and y.
- [ ] ***system.py***: Replace the hardcoded weights of scores.