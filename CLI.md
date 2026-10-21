# Training from the terminal

From this directory, run:

```bash
python3 run_training.py
```

There are no required arguments. Data is read from `train.jsonl`,
`validation.jsonl`, and `test.jsonl` alongside the script, even if you launch
it from another directory. Every run initializes fresh random weights using a
fixed seed for reproducibility, then replaces outputs in `results/`.

Activate the environment described in README.md before running this command.
Only NumPy is required.

## What the display means

Tables update in place in a terminal. Redirected output is plain text and
prints every ten epochs. Lower binary cross entropy (BCE) loss means the
probabilities fit the labels better; accuracy is the fraction of correct
noun/verb decisions at the 50% threshold.

The confusion matrix appears **only at the end**, using held-out test data
and the selected best checkpoint. Rows are the true labels; columns are the
predicted labels:

| Actual / predicted | noun | verb |
| --- | --- | --- |
| noun | Correct noun | Noun incorrectly called verb |
| verb | Verb incorrectly called noun | Correct verb |

If we call **verb** the positive class, the upper-right cell is false positives
and the lower-left cell is false negatives. The diagonal counts correct
predictions. The best checkpoint is chosen by the lowest validation loss,
which need not be the epoch with the highest accuracy.

**Test** results appear only after training ends and the best checkpoint has
been selected. No confusion matrix is displayed during training. Training never uses test
results to choose a checkpoint.

## Demo pacing and outputs

`run_training.py` contains presentation settings at the top: a 0.10-second
pause per displayed frame and at most about 100 regular refreshes for a
300-epoch budget. Early stopping can end sooner. Model updates continue at
full speed between refreshes. The compute time excludes callback rendering
and sleep; the display pause time is reported separately. Redirected output
has no artificial sleep.

Ctrl-C ends the run without a Python traceback. The renderer never hides the
cursor or changes terminal modes. Any already saved best checkpoint remains;
a run interrupted before its first checkpoint may leave a checkpoint from a
previous run, so rerun training to completion before presenting final results.

Completed training writes:

- `results/best_model.npz`: selected weights and vocabulary for inference.
- `results/history.json`: training and validation learning curves.
- `results/report.json`: configuration, timings, best epoch, and final metrics.
- `results/predictions.jsonl`: predictions for inspection.
- `results/one_update.json`: one illustrative SGD update, including gradients
  and before/after values. Full training uses Adam.

These are synthetic, controlled examples. Held-out sentence families are a
useful demonstration of limited generalization, not evidence of broad English
understanding. See `TRAINING.md` for the numerical checks and measured results.
