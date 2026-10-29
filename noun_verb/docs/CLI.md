# Training from the terminal

Follow [setup](../README.md#setup), then run from the project folder:

```bash
python run_training.py
```

Each run starts fresh and replaces outputs in `results/`. Data paths are relative to the script.

## Display

The table shows epoch, training/validation loss and accuracy, and best epoch. Lower loss means better-fitting probabilities; accuracy counts correct noun, verb, or other predictions using the most probable label.

Validation loss selects the checkpoint. Test results appear after training, with confusion-matrix rows showing actual labels and columns showing predictions. Diagonal entries are correct.

Adjust `DISPLAY_DELAY_SECONDS`, `MAX_REFRESHES`, and `MAX_EPOCHS` in `run_training.py`. Compute time excludes display pauses. Redirected output prints every ten epochs without pauses.

Ctrl-C stops training. Complete a new run before using final results; an interrupted run may leave older weights.

## Outputs

- `best_model.npz`: selected weights and vocabulary.
- `history.json`: training and validation curves.
- `report.json`: configuration, timing, and metrics.
- `predictions.jsonl`: training, validation, and test predictions.
- `one_update.json`: illustrative SGD update; full training uses Adam.

All outputs live in `results/`. See [training](TRAINING.md) and [prediction](PREDICTION.md).
