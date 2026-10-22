# Noun or verb: a tiny learning demo

Train a small neural network from scratch to classify one selected word using its context. Requires Python 3.10+ and NumPy.

## Setup

Run from this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train and predict

```bash
python run_training.py
python predict.py
```

In `predict.py`, enter a sentence, then select a word or displayed token number. Try “They watch the birds.” and “The watch is broken.” Submit an empty sentence to exit.

For a single prediction:

```bash
python inference.py watch "They watch the birds."
```

## Explore

- Complete the two TODOs in `student_exercise.py`.
- Check the [answer key](docs/STUDENT_EXERCISE_ANSWERS.md).
- Run `python example_inference.py` for the Python API.
- Run `python check_model.py` to verify the model calculations.
- Read [training](docs/TRAINING.md), [prediction](docs/PREDICTION.md), and [terminal usage](docs/CLI.md).

This synthetic demo supports noun/verb decisions only. High confidence does not guarantee correctness.
