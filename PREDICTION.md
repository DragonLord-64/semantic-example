# Classify a selected word with saved weights

From this folder, activate the prepared environment and pass the word to classify
and then a quoted sentence:

```bash
source .venv/bin/activate
python inference.py watch "They watch the birds."
python inference.py watch "The watch is broken."
```

No input file is needed. The script loads `results/best_model.npz` relative to
its own folder and never trains or modifies the checkpoint. Missing weights
produce the training command. For a fresh environment on another machine:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

This is a **binary classifier for one selected occurrence**, not a word
extractor or full sentence tagger. It always chooses noun or verb. It cannot
identify other grammatical roles. Choose an occurrence that you expect to be
one of those two roles.

For repeated words, supply a **zero-based token index** instead of a word:

```bash
python inference.py 1 "They watch the watch."
python inference.py 3 "They watch the watch."
```

These select the first and second occurrences of watch respectively. A repeated
word supplied by name produces a message listing the available zero-based
indices. The CLI display labels selected positions starting at 1; the selection
argument and Python API indices start at 0.

The display shows noun/verb probabilities and the two words on each side of
the selection. `<PAD>` marks boundaries. Unknown words use `<UNK>` and are
reported. Capitalization and punctuation are discarded with the training
tokenizer. Probabilities are not a correctness guarantee.

You can select any word. The model was supervised only on watch, play, call,
walk, run, drink, kick, park, plant, dance, brush, wave, smile, answer, dream,
and light. Selecting another word produces a training-coverage warning,
even if that word appeared in context during training. Unseen target words
also receive the unknown-word warning.

## Import from another Python script

Run `python example_inference.py`, or use:

```python
from inference import Predictor, load_model

predictor = Predictor.from_checkpoint()  # Load once; no training.
result = predictor.predict_word("They watch the birds.", "watch")
result = predictor.predict("They watch the watch.", target_index=3)

# Alternatively, supply a loaded model yourself:
model = load_model()  # Optional checkpoint path accepted here and above.
predictor = Predictor(model)
```

Imports do not prompt, load checkpoints, or train. Put your script in this
folder or add this folder to its Python module path. Checkpoint defaults remain
relative to `inference.py`, independent of your current working directory.
Returns are plain dictionaries containing prediction, noun/verb probabilities,
context, target index, training-coverage and unknown-word information.

`target_index` is zero-based over `model.tokenize(sentence)`, not whitespace
positions. Invalid indices or missing/ambiguous target words raise ValueError.

An optional `predictor.predict_sentence(sentence)` helper evaluates occurrences
of the sixteen known training targets only and marks other tokens unsupported.
It is not an all-word tagger. Explicit `predict`/`predict_word` calls above are
the primary API and allow any selected word.

The earlier interactive entry point remains available as `python predict.py`.
Use `:quit`, an empty sentence, Ctrl-D, or Ctrl-C to finish.
