"""Download the same model used by search.py into the shared local cache."""

from fastembed import TextEmbedding

if __package__:
    from .search import CACHE_DIR, MODEL_NAME
else:
    from search import CACHE_DIR, MODEL_NAME


if __name__ == "__main__":
    print(f"Downloading {MODEL_NAME} to {CACHE_DIR}...", flush=True)
    model = TextEmbedding(model_name=MODEL_NAME, cache_dir=str(CACHE_DIR))
    vector = next(iter(model.query_embed("Model setup check")))
    print(f"Ready: generated an embedding with {len(vector)} dimensions.")
