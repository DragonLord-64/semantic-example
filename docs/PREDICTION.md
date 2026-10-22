# Prediction

Follow [setup](../README.md#setup) and train once. Run commands from the project folder; predictions load `results/best_model.npz` without retraining.

## Interactive

```bash
python predict.py
```

Enter a sentence, then a word or displayed **one-based** token number. Results show the label and its probability. Use `:back` for another sentence; `:quit` or an empty sentence exits.

## Single command

Pass the word first, then a quoted sentence:

```bash
python inference.py watch "They watch the birds."
python inference.py watch "The watch is broken."
```

Repeated words require a **zero-based** token index:

```bash
python inference.py 3 "They watch the watch."
```

This selects the second “watch”. Indices follow `model.tokenize(sentence)`.

## Python API

```python
from inference import Predictor

predictor = Predictor.from_checkpoint()
result = predictor.predict_word("They watch the birds.", "watch")
print(result["prediction"], result["verb_probability"])

# Select a repeated word by zero-based index:
result = predictor.predict("They watch the watch.", target_index=3)
```

Reuse the predictor across sentences. Missing or ambiguous words and invalid indices raise `ValueError`. See `example_inference.py` for another example.

Only noun and verb are supported. Unknown words share one vector; probabilities do not guarantee correctness.
