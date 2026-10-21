# Tiny noun/verb classifier

This implementation learns from scratch. It predicts **the role of a selected word**,
not tags for the whole sentence. Dataset records use a zero-based token index;
the interactive prediction tool displays one-based word numbers for the user.
There is no pretrained vocabulary, embedding, or language model.

## Run

Activate the environment from README.md, then run:

```bash
python check_model.py
python run_training.py
python inference.py watch "They watch the birds."
```

The CLI takes the word first and the sentence second. See `PREDICTION.md` for the
importable API and repeated-word selection.
No GPU or downloads of model weights are required. Rerunning `train.py` overwrites
the selected output directory; use `--output-dir another-directory` to preserve a run.

## The calculation

1. Lowercase and tokenize the sentence, using the dataset's token pattern.
2. Take two words before the target, the target, and two words after it.
3. Look up an eight-number learned embedding for each token.
4. Flatten the five vectors into 40 ordered inputs.
5. Compute `h = tanh(x @ W1 + b1)` with 16 hidden neurons.
6. Compute `z = h @ W2 + b2`, then `P(verb) = sigmoid(z)`.
7. Predict verb at probability >= 0.5; otherwise noun.

The label is 1 for verb, 0 for noun. Binary cross entropy is computed stably as
`logaddexp(0, z) - y*z`. Its logit derivative is `sigmoid(z) - y`.
`model.py` shows the complete derivatives, including accumulating embedding gradients
when the same word occurs more than once in a batch. All parameters, including
embeddings and the padding vector, are trainable. Padding carries boundary information.

The vocabulary is built **only from training sentences**. Missing words become
`<UNK>` and sentence boundaries become `<PAD>`. Ten percent of non-padding context
tokens (not the target) are replaced by `<UNK>` during training so that this vector
receives training. Positions remain separate; no bag-of-words averaging is used.

With this dataset: 259 vocabulary entries, `8*259 + 673 = 2,745` parameters.

## Training and selection

The fixed configuration uses seed 7, Adam with learning rate 0.02, batches of 32,
at most 300 epochs, and early stopping after 30 epochs without validation-loss
improvement of at least 1e-6. This configuration was chosen before the first test
evaluation; it has not been tuned on test results. Training labels update weights;
validation loss selects the checkpoint. Only after selection does the script open
the test split, which is evaluated once per run.

Measured run on this host:

| Split | Examples | Accuracy | Cross entropy |
| --- | ---: | ---: | ---: |
| Training | 192 | 100% | 0.0009 |
| Validation | 64 | 84.375% | 0.2617 |
| Test | 64 | 96.875% (62/64) | 0.1653 |

The training loop took approximately **0.14 seconds**, excluding Python startup,
imports, and report writing. Epoch 4 was selected; training stopped at epoch 34.
The initial training loss was approximately 0.71. The constant-noun baseline is
50%; every target has equal noun/verb examples, so target identity alone provides
no class information.

## Evidence and limitations

`check_model.py` checks **every entry of every parameter tensor** against central
finite differences, with repeated token IDs included. Maximum absolute gradient
error was 1.03e-11. It also checks a loss-reducing update, exact prediction equality
after serialization, and finite losses/gradients for logits near +/-1000.

The test confusion matrix (actual rows, predicted columns; noun then verb) is
`[[32, 0], [2, 30]]`. The two missed examples are “Do you drink milk?” and
“Do you plant vegetables?”. Test question verbs scored 14/16; the other three
test template families scored 16/16. Validation is harder for this model than
test: it misses 4/16 infinitives and 6/16 negated verbs. Different split difficulties
explain why test accuracy exceeds validation accuracy; neither score estimates
general English performance.

These are only 320 synthetic sentences from a few repeated templates. Family
separation makes evaluation better than randomly splitting near-duplicates, but
sentences and errors still share structure and are not independent natural-language
samples. Easy article/pronoun and position cues do much of the work. All 16 target
words appear in training; there is no evidence of generalization to unseen targets.
Test context OOV rate is 58.0% (excluding padding and target); 62/64 examples contain
at least one unknown context word. Unknown words share one vector, so the model
cannot distinguish their meanings. Outputs are model scores, not calibrated certainty.
This is supervised grammatical classification, not semantic search.

## Files for the future visualization

- `results/history.json`: epoch 0 and every epoch's train/validation loss and accuracy.
- `results/best_model.npz`: selected weights and exact training vocabulary.
- `results/report.json`: configuration, timing, confusion matrices, per-target and
  per-template accuracy, and OOV rates.
- `results/predictions.jsonl`: sentences, true labels, predicted probabilities,
  correctness and encoded windows for the selected checkpoint.
- `results/one_update.json`: one illustrative plain SGD step from initialization,
  including full parameters before/after and gradients. This is separate from the
  Adam run and explicitly labeled. It can explain “error nudges the weights”.

`train.train(...)` accepts an optional `callback(event)` for terminal progress.
Epoch events contain epoch, train/validation loss and accuracy, validation confusion,
learning rate, best epoch and computation seconds. Completion includes the report.
Callback display/pacing time is excluded from reported training computation time.
