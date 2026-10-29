# Training

Follow [setup](../README.md#setup), then run from the project folder:

```bash
python run_training.py
```

## Model

The model learns from scratch:

1. Lowercase and tokenize the sentence.
2. Select the target and two words on each side.
3. Join five learned eight-number embeddings into 40 inputs.
4. Apply 16 hidden neurons: `h = tanh(x @ W1 + b1)`.
5. Compute three logits, then `softmax(h @ W2 + b2)` for noun, verb, and other probabilities.
6. Predict the label with the highest probability.

Training uses categorical cross entropy and Adam. Vocabulary comes from training sentences; `<UNK>` represents unseen words and `<PAD>` marks boundaries. Context dropout replaces 10% of non-padding context tokens with `<UNK>`.

## Selection and checks

Defaults: seed 7, learning rate 0.02, batches of 32, up to 300 epochs, and early stopping after 30 epochs without validation-loss improvement. Training updates weights; validation loss selects the checkpoint; test evaluation follows selection.

The synthetic dataset contains noun, verb, and other examples. `other` includes roles such as pronouns, determiners, and prepositions, allowing the predictor to process every token in the student exercise. See [manifest.json](../manifest.json) for current counts and split details. Results demonstrate limited generalization, not broad English understanding.

Run `python check_model.py` for numerical gradients, a loss-reducing update, save/load consistency, and extreme-logit stability. Current metrics are in [report.json](../results/report.json).

Change training defaults in `train.py`, model dimensions in `model.py`, and the dashboard epoch limit in `run_training.py`. Retrain after changes. See [terminal usage](CLI.md) for outputs and display settings.
