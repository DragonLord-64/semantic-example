# Noun, verb, or other: a tiny learning demo

For a pretrained model that searches a text file by meaning, see the separate [semantic search example](semantic_search/README.md). Edit the file path and query in its Python script, then run it directly in VS Code.

Train a small neural network from scratch to classify one selected word as noun, verb, or other using its context. Requires Python 3.10+ and NumPy.

## Setup

With Docker and VS Code's Dev Containers extension installed, choose **Dev Containers: Reopen in Container**. Dependencies install automatically; then run the commands below under “Train and predict”. [Dev Containers setup](https://code.visualstudio.com/docs/devcontainers/containers).

For local setup:

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

- Start with [student-example](noun_verb/student_example.py): `python noun_verb/student_example.py` shows how to load the module and predict a word's label in different sentences.
- Complete [student_exercise.py](noun_verb/student_exercise.py): create a function that finds the nouns and verbs in “They watch the birds.”
- Run the [answer key](noun_verb/student_exercise_answers.py): `python noun_verb/student_exercise_answers.py`.
- Run `python example_inference.py` for the Python API.
- Run `python check_model.py` to verify the model calculations.
- Run `python -m unittest discover -s tests -v` for the automated tests.
- Read [training](docs/TRAINING.md), [prediction](docs/PREDICTION.md), and [terminal usage](docs/CLI.md).

The exercise's reference solution tokenizes the sentence, predicts each word in its original sentence context, and collects the nouns and verbs while skipping `other` predictions. It uses token indices so repeated words work too.

This synthetic demo supports noun, verb, and other labels. `other` covers words such as pronouns, determiners, and prepositions. It does not identify their individual parts of speech. High confidence does not guarantee correctness.
