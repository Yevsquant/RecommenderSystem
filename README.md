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
- [ ] ***item_cf.py***: Require a more complex metric to compute similarity of items. Currently the similarity is computed via Co-Occurrence.
- [ ] ***item_cf.py***: Implement a index database for user2item like matrix, item2item similarity matrix, and user2item recommend matrix.