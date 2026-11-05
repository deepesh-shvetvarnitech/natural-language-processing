# Day 4 — Embeddings & Semantic Search

A beginner-friendly project that compares lexical search (TF-IDF) with semantic search (Sentence Transformers) on product reviews.

## Project structure
```text
day04_embeddings_semantic_search/
├── README.md
├── requirements.txt
├── data/
│   ├── product_reviews.csv
│   └── clean_reviews.csv            # generated
├── src/
│   ├── __init__.py
│   ├── api.py
│   ├── embeddings.py
│   ├── evaluate.py
│   ├── main.py
│   ├── make_dataset.py
│   ├── preprocess.py
│   └── search.py
├── artifacts/                       # generated
│   ├── review_embeddings.npy
│   ├── tfidf_matrix.npz
│   ├── tfidf_vectorizer.joblib
│   └── embedding_model_name.txt
├── outputs/                          # generated
│   └── vocabulary.json
├── reports/                          # generated
│   └── evaluation.md
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_api.py
    ├── test_embeddings.py
    └── test_search.py
```

## Setup (Windows PowerShell)
Run commands from this project directory:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run in order
```powershell
python -m src.make_dataset
python -m src.main
python -m src.evaluate
pytest -q
uvicorn src.api:app --reload
```

Open `http://127.0.0.1:8000/docs` for Swagger.

### API examples
`GET /health`

`POST /similar-reviews`
```json
{
  "query": "battery lasts all day and charges quickly",
  "top_k": 5,
  "search_type": "semantic"
}
```
Set `search_type` to `tfidf` to compare lexical search. `exclude_review_id` can omit a known source review.

## Notes
- The first Sentence Transformers run downloads `sentence-transformers/all-MiniLM-L6-v2`; internet is required once.
- Corpus reviews and search queries must use the same embedding model.
- `product_reviews.csv` is a small included demonstration dataset. Replace it with your assignment dataset while preserving `review_id`, `review`, and `category`.
- Evaluation uses categories as relevance labels. It is a simple demonstration, not a universal quality judgment.
