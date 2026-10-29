"""Reference solution."""

from pathlib import Path

import numpy as np

if __package__:
    from .student_exercise import run
else:
    from student_exercise import run


def search(folder, query, model, k=3):
    if not query.strip():
        raise ValueError("The query must not be empty.")
    if k < 1:
        raise ValueError("k must be at least 1.")
    folder = Path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(folder)

    rows = []
    for path in sorted(folder.glob("*.txt")):
        if not path.is_file():
            continue
        for n, text in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if text.strip():
                rows.append({"file": path.name, "line": n, "text": text})
    if not rows:
        return []

    vecs = np.stack(list(model.passage_embed([r["text"] for r in rows])))
    q = next(iter(model.query_embed(query)))

    norms = np.linalg.norm(vecs, axis=1) * np.linalg.norm(q)
    scores = np.divide(
        vecs @ q,
        norms,
        out=np.zeros(len(rows), dtype=float),
        where=norms != 0,
    )
    best = np.argsort(-scores, kind="stable")[:k]
    return [{**rows[i], "score": float(scores[i])} for i in best]


if __name__ == "__main__":
    run(search)
