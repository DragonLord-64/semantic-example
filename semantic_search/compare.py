"""Edit the two strings below, then press Run Python File in VS Code."""

from pathlib import Path

import numpy as np
from fastembed import TextEmbedding

TEXT_1 = "I forgot my password."
TEXT_2 = "I cannot remember my login password."


def main():
    model = TextEmbedding(
        model_name="BAAI/bge-small-en-v1.5",
        cache_dir=str(Path(__file__).resolve().parent / ".model_cache"),
    )

    # Turn each string into a vector of numbers representing its meaning.
    vector_1, vector_2 = list(model.embed([TEXT_1, TEXT_2]))

    # Cosine similarity compares the directions of the two vectors.
    dot_product = np.dot(vector_1, vector_2)
    length_1 = np.linalg.norm(vector_1)
    length_2 = np.linalg.norm(vector_2)
    similarity = dot_product / (length_1 * length_2)

    print(f"Text 1: {TEXT_1}")
    print(f"Text 2: {TEXT_2}")
    print(f"Similarity: {similarity:.3f} (higher means more similar)")


if __name__ == "__main__":
    main()
