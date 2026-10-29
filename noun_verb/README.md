# Noun, verb, or other: a tiny learning demo

Train a small neural network from scratch to classify one selected word as noun, verb, or other using its context. Requires Python 3.10+ and NumPy.

## Setup

With Docker and VS Code's Dev Containers extension installed, choose **Dev Containers: Reopen in Container**. Dependencies install automatically; then run the commands below under “Train and predict”. [Dev Containers setup](https://code.visualstudio.com/docs/devcontainers/containers).

For local setup:

Run from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r noun_verb/requirements.txt
cd noun_verb
```

## Train and predict

Run these commands from `noun_verb/` (use `cd noun_verb` after opening the dev container).

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

- Start with [student-example](student_example.py): `python student_example.py` shows how to load the module and predict a word's label in different sentences.
- Complete [student_exercise.py](student_exercise.py): create a function that finds the nouns and verbs in “They watch the birds.”
- Run the [answer key](student_exercise_answers.py): `python student_exercise_answers.py`.
- Run `python example_inference.py` for the Python API.
- Run `python check_model.py` to verify the model calculations.
- Run `cd .. && python -m unittest discover -s noun_verb/tests -v` for the automated tests.
- Read [training](docs/TRAINING.md), [prediction](docs/PREDICTION.md), and [terminal usage](docs/CLI.md).

The exercise's reference solution tokenizes the sentence, predicts each word in its original sentence context, and collects the nouns and verbs while skipping `other` predictions. It uses token indices so repeated words work too.

This synthetic demo supports noun, verb, and other labels. `other` covers words such as pronouns, determiners, and prepositions. It does not identify their individual parts of speech. High confidence does not guarantee correctness.
