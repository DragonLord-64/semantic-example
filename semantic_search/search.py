"""Edit FILE_PATH and QUERY below, then press Run Python File in VS Code."""

from pathlib import Path

import numpy as np
from fastembed import TextEmbedding

# Change these two settings to try your own text file and question.
# Relative paths are resolved beside this script, regardless of working directory.
FILE_PATH = "sample.txt"
QUERY = "How do I recover a forgotten password?"
TOP_K = 3

MODEL_NAME = "BAAI/bge-small-en-v1.5"
CACHE_DIR = Path(__file__).resolve().parent / ".model_cache"


def search_file(file_path, query, model, top_k=3):
    """Return (line number, original text, cosine similarity), best first."""
    if not query.strip():
        raise ValueError("The query must not be empty.")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    lines = [
        (number, text)
        for number, text in enumerate(
            Path(file_path).read_text(encoding="utf-8").splitlines(), start=1
        )
        if text.strip()
    ]
    if not lines:
        return []

    # Each non-empty line gets its own vector. Keep demo lines short:
    # this model truncates input beyond its 512-token context window.
    vectors = np.stack(list(model.passage_embed([text for _, text in lines])))
    query_vector = next(iter(model.query_embed(query)))

    # Cosine similarity = dot product / (length of vector A * length of vector B).
    denominators = np.linalg.norm(vectors, axis=1) * np.linalg.norm(query_vector)
    scores = np.divide(
        vectors @ query_vector,
        denominators,
        out=np.zeros(len(lines), dtype=float),
        where=denominators != 0,
    )
    best = np.argsort(-scores, kind="stable")[:top_k]
    return [(lines[i][0], lines[i][1], float(scores[i])) for i in best]


def main():
    path = Path(FILE_PATH).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path
    if not path.is_file():
        raise FileNotFoundError(f"Set FILE_PATH to an existing text file: {path}")

    print(f"File: {path}\nQuery: {QUERY}\n", flush=True)
    print("Loading embedding model (first use downloads it)...", flush=True)
    model = TextEmbedding(model_name=MODEL_NAME, cache_dir=str(CACHE_DIR))
    matches = search_file(path, QUERY, model, TOP_K)
    if not matches:
        print("The file has no non-empty lines to search.")
        return
    print("\nTop matches:")
    for rank, (line_number, text, score) in enumerate(matches, start=1):
        print(f"{rank}. Line {line_number} | Similarity: {score:.3f}\n   {text}\n")


if __name__ == "__main__":
    main()
