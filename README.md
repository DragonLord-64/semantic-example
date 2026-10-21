# Noun or verb: a tiny learning demo

Train a small neural network from scratch, then classify a selected word using its sentence context. This is a controlled teaching example, not a general part-of-speech tagger or semantic search engine.

## Setup

Use Python 3.10 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run commands from this folder. Only NumPy is required; no GPU or pretrained model is used.

## Train

```bash
python run_training.py
```

No arguments are needed. Paths are relative to the scripts, so training also works from another directory. The terminal table updates in place with epoch, training and validation loss, accuracy, learning rate, and best epoch. Short display pauses make progress visible. Redirected output uses plain text. The confusion matrix appears only at the end, with actual labels as rows and predicted labels as columns.

Training starts with random weights. Validation loss selects the checkpoint; the test set is evaluated after selection. Saved weights live in `results/best_model.npz`; JSON files contain metrics, history, predictions, and one illustrative weight update.

## Predict from the command line

Pass the **word first**, followed by one quoted sentence:

```bash
python inference.py watch "They watch the birds."
python inference.py watch "The watch is broken."
```

For repeated words, pass a zero-based token index instead:

```bash
python inference.py 3 "They watch the watch."
```

## Call from Python

```python
from inference import Predictor

predictor = Predictor.from_checkpoint()  # Load saved weights once.
result = predictor.predict_word("They watch the birds.", "watch")
print(result["prediction"], result["verb_probability"])
```

Run `python example_inference.py` for another example. Importing the module does not train or prompt. Use `predict(sentence, target_index)` for an explicit occurrence.

For the student exercise, open `student_exercise.py`. Finish two TODOs: load the saved model, then loop over sentences containing “watch” and print its predicted role. Compare with your own answers, including a deliberately tricky short sentence.

## Tweak and verify

In `train.py`, change the `train()` defaults for learning rate, batch size, patience, and context dropout. In `run_training.py`, change `MAX_EPOCHS` and `DISPLAY_DELAY_SECONDS`. Embedding and hidden dimensions are defined in `model.py`. Retrain after changes; checkpoint loading restores saved dimensions.

Run `python check_model.py` for numerical gradients, save/load consistency, and stable extreme-logit calculations.

The dataset contains 320 sentences: 192 training, 64 validation, and 64 test. Sixteen ambiguous target words appear equally as nouns and verbs; template families are separated across splits. Any selected word is accepted, but unseen words share an unknown representation. Context can help; high confidence does not establish accuracy. There is no “other” class, so select a word that is a noun or verb.
