"""Search non-empty lines in a folder's UTF-8 .txt files using the given model
and cosine similarity, keeping embeddings in memory.
Return the top k matches across all files as dictionaries with file, line
(original one-based number), text, and score.
"""

from pathlib import Path

import numpy as np
from fastembed import TextEmbedding

DIR = "sample_documents"
QUERY = "How do I recover a forgotten password?"
K = 3

MODEL = "BAAI/bge-small-en-v1.5"
CACHE = Path(__file__).resolve().parent / ".model_cache"


def search(folder, query, model, k=3):
    # TODO: Implement search.
    raise NotImplementedError


def run(fn):
    folder = Path(DIR).expanduser()
    if not folder.is_absolute():
        folder = Path(__file__).resolve().parent / folder
    model = TextEmbedding(model_name=MODEL, cache_dir=str(CACHE))
    hits = fn(folder, QUERY, model, K)
    if not hits:
        print("No matches.")
        return
    for i, hit in enumerate(hits, start=1):
        print(
            f"{i}. {hit['file']}:{hit['line']} ({hit['score']:.3f})\n"
            f"   {hit['text']}"
        )


if __name__ == "__main__":
    run(search)
