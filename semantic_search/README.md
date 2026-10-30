# Search a file by meaning

## Simplest example: compare two strings

After running setup below, open `compare.py`, edit `TEXT_1` and `TEXT_2`, and press **Run Python File**. Or run it from the repository root:

```bash
.venv/bin/python semantic_search/compare.py
```

It turns the two strings into embeddings and prints their cosine similarity. Higher scores mean more similar meanings; the score is not a probability or a percentage. Try replacing the second string with `"The garden needs watering."` and compare the score.

## Setup and file search

Run setup once from the repository root (internet required for installation and model download):

```bash
bash semantic_search/setup.sh
```

In VS Code, use **Python: Select Interpreter** to select the interpreter printed by setup. Open `search.py`, edit `FILE_PATH` and `QUERY` at the top, and press **Run Python File**. No command-line arguments are needed. You can also run `.venv/bin/python semantic_search/search.py` when using the repository's virtual environment.

`FILE_PATH` can be an absolute path or a path relative to `search.py`. Use a UTF-8 text file with one short sentence or passage per line. Blank lines are skipped; printed line numbers still match the original file. Keep lines short: the model supports up to 512 tokens per input and truncates longer lines.

The included `sample.txt` has twenty sentences grouped into four concepts: account access (lines 1–5), baking (6–10), gardening (11–15), and train travel (16–20). Try questions about each topic and compare the rankings.

The script embeds each non-empty line with FastEmbed's `passage_embed`, embeds your question with `query_embed`, and ranks lines by cosine similarity. Higher scores mean closer matches, not probabilities. Change `TOP_K` to show more results. Every run embeds the file again; for this small demo the vectors live in memory.

The pretrained English model runs locally on CPU and is cached in `.model_cache/`. After setup, repeated runs reuse the downloaded model. No API key or database is required.

For use from Python, import `search_file` from `semantic_search.search` and pass a file path, query, and a `fastembed.TextEmbedding` instance. Reuse the model instance for multiple searches.

Model and API reference: [FastEmbed](https://qdrant.github.io/fastembed/Getting%20Started/).

## Student exercise: search a directory

Implement `search` in `student_exercise.py` using `search.py` as a reference. Edit `DIR`, `QUERY`, and `K`, then press **Run Python File**.

Sample files are in `sample_documents/`; the solution is in `student_exercise_answers.py`.
