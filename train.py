"""Train on train, select by validation loss, then evaluate test once."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import time
import numpy as np
from model import Adam, ContextModel, UNK, PAD, LABELS, build_vocab, cross_entropy, encode, softmax


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def evaluate(model, rows):
    # Measure a fixed model: this function never updates its weights. Each row
    # supplies a sentence, a target position, and the correct noun/verb/other label.
    windows, labels = encode(rows, model.vocab)
    # Forward pass: word embeddings -> hidden layer -> three scores (logits).
    # Softmax turns them into probabilities, ordered noun, verb, other.
    logits = model.forward(windows)[0]
    probabilities = softmax(logits)
    predicted = probabilities.argmax(axis=1)
    correct = predicted == labels
    confusion = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    # Rows are actual labels; columns are predictions. Both use LABELS.
    np.add.at(confusion, (labels.astype(int), predicted.astype(int)), 1)
    context_mask = np.ones_like(windows, dtype=bool)
    context_mask[:, 2] = False
    context_mask &= windows != model.vocab[PAD]
    oov = (windows == model.vocab[UNK]) & context_mask
    groups = {}
    # These breakdowns expose weak words/patterns and unfamiliar context words
    # (OOV = out of vocabulary), which a single accuracy score can hide.
    for field in ['target', 'template_family']:
        grouped = defaultdict(list)
        for i, row in enumerate(rows):
            grouped[row[field]].append(i)
        groups['by_' + field] = {
            key: {'count': len(indices), 'accuracy': float(correct[indices].mean()),
                  'context_oov_rate': float(oov[indices].sum() / max(1, context_mask[indices].sum())),
                  'examples_with_oov_context': int(oov[indices].any(axis=1).sum())}
            for key, indices in sorted(grouped.items())}
    return {
        'count': len(rows),
        # Multiclass cross-entropy penalizes confident wrong answers.
        'loss': cross_entropy(logits, labels),
        'accuracy': float(correct.mean()),
        'confusion_rows_actual_columns_predicted_noun_verb_other': confusion.tolist(),
        'context_oov_rate': float(oov.sum() / max(1, context_mask.sum())),
        'examples_with_oov_context': int(oov.any(axis=1).sum()),
        **groups,
    }, [dict(row, probabilities=dict(zip(LABELS, probabilities[i].tolist())),
             prediction=LABELS[predicted[i]], correct=bool(correct[i]),
             window_ids=windows[i].tolist()) for i, row in enumerate(rows)]


def train(data_dir, output_dir, seed=7, max_epochs=300, patience=30,
          learning_rate=0.02, batch_size=32, context_dropout=0.1,
          callback=None, verbose=True):
    # 1. Load examples. Only training rows build the vocabulary and update
    # weights. Validation chooses the checkpoint; test stays closed until then.
    output_dir.mkdir(parents=True, exist_ok=True)
    training = read_rows(data_dir / 'train.jsonl')
    validation = read_rows(data_dir / 'validation.jsonl')
    vocab = build_vocab(training)
    # Start from scratch, with reproducible random embeddings and layer weights
    # (biases start at zero). Both embeddings and layer weights will learn.
    model = ContextModel(vocab, seed=seed)
    model.targets = frozenset(row['target'] for row in training)
    # Each input keeps five ordered positions: two left, target, two right.
    # encode() uses word IDs, PAD at sentence edges, and UNK for unseen words.
    # The model looks up 8 numbers per ID and concatenates them in this order,
    # then uses 16 tanh hidden units to produce its noun/verb/other prediction.
    windows, labels = encode(training, vocab)
    optimizer = Adam(model.params, learning_rate)
    rng = np.random.default_rng(seed + 1)
    # Record the untrained baseline (epoch 0), before any real training updates.
    initial, _ = evaluate(model, training)
    initial_validation, _ = evaluate(model, validation)
    history = [dict(epoch=0, train_loss=initial['loss'], train_accuracy=initial['accuracy'],
                    validation_loss=initial_validation['loss'],
                    validation_accuracy=initial_validation['accuracy'])]
    # 2. Save an inspectable example of learning: predict, calculate loss and
    # gradients by backpropagation, then subtract learning_rate * gradient.
    # This is one illustrative SGD step; the full run below uses Adam instead.
    one_x, one_y = windows[:1], labels[:1]
    before = dict(zip(LABELS, model.predict(one_x)[0].tolist()))
    loss, gradients = model.loss_and_gradients(one_x, one_y)
    original = {key: value.copy() for key, value in model.params.items()}
    for key in model.params:
        model.params[key] -= 0.1 * gradients[key]
    after = dict(zip(LABELS, model.predict(one_x)[0].tolist()))
    trace = {'example': training[0], 'window_ids': one_x[0].tolist(),
             'probabilities_before': before, 'loss_before': loss,
             'probabilities_after': after,
             'loss_after': model.loss_and_gradients(one_x, one_y)[0],
             'optimizer': 'one illustrative SGD step; full run uses Adam',
             'learning_rate': 0.1,
             'parameters_before': {k: v.tolist() for k, v in original.items()},
             'gradients': {k: v.tolist() for k, v in gradients.items()},
             'parameters_after': {k: v.tolist() for k, v in model.params.items()}}
    model.params = original
    # Restore the original parameters, so this teaching trace does not give
    # the actual training run an extra update or change its initial baseline.
    (output_dir / 'one_update.json').write_text(json.dumps(trace, indent=2) + '\n')
    best_loss, best_epoch, stale = float('inf'), 0, 0
    callback_seconds = 0.
    started = time.perf_counter()

    def emit_epoch(metrics):
        nonlocal callback_seconds
        if callback is not None:
            event = dict(history[-1], type='epoch',
                         validation_confusion=metrics['confusion_rows_actual_columns_predicted_noun_verb_other'],
                         learning_rate=learning_rate, best_epoch=best_epoch,
                         compute_seconds=time.perf_counter() - started - callback_seconds)
            callback_started = time.perf_counter()
            callback(event)
            # The CLI callback may pause to make updates visible. Exclude that
            # display time from training_seconds; this is not total wall time.
            callback_seconds += time.perf_counter() - callback_started

    emit_epoch(initial_validation)
    # 3. An epoch is one pass over all training rows. Shuffle every epoch and
    # split into minibatches, averaging each batch's error before an update.
    for epoch in range(1, max_epochs + 1):
        for indices in np.array_split(rng.permutation(len(labels)),
                                      int(np.ceil(len(labels) / batch_size))):
            batch = windows[indices].copy()
            # Training-only context dropout replaces some neighboring words
            # with UNK, practicing missing/unfamiliar context. Keep the target
            # and sentence-edge PAD intact. Evaluation uses the original rows.
            mask = rng.random(batch.shape) < context_dropout
            mask[:, 2] = False
            mask &= batch != vocab[PAD]
            batch[mask] = vocab[UNK]
            # Forward pass measures error; backpropagation calculates how each
            # parameter affects it. Adam uses running gradient statistics to
            # adjust embeddings, layer weights, and biases together.
            _, gradients = model.loss_and_gradients(batch, labels[indices])
            optimizer.update(model.params, gradients)
        # Evaluate without dropout or weight updates. Training accuracy measures
        # fit to examples seen; validation measures performance on held-out rows.
        train_metrics, _ = evaluate(model, training)
        val_metrics, _ = evaluate(model, validation)
        history.append(dict(epoch=epoch, train_loss=train_metrics['loss'],
                            train_accuracy=train_metrics['accuracy'],
                            validation_loss=val_metrics['loss'],
                            validation_accuracy=val_metrics['accuracy']))
        if val_metrics['loss'] < best_loss - 1e-6:
            # Keep the lowest validation loss, not necessarily the last epoch
            # or the highest training accuracy. Test results never choose this.
            best_loss, best_epoch, stale = val_metrics['loss'], epoch, 0
            model.save(output_dir / 'best_model.npz')
        else:
            stale += 1
        emit_epoch(val_metrics)
        if stale >= patience:
            # Early stopping: stop after this many epochs without improvement.
            break
    elapsed = time.perf_counter() - started - callback_seconds
    model = ContextModel.load(output_dir / 'best_model.npz')
    # 4. Restore the selected weights, then open test for the final evaluation.
    # Held-out test results describe this toy dataset, not general English.
    test = read_rows(data_dir / 'test.jsonl')
    report = {'seed': seed, 'vocabulary_size': len(vocab),
              'parameter_count': sum(p.size for p in model.params.values()),
              'training_seconds': elapsed, 'epochs_run': epoch, 'best_epoch': best_epoch,
              'configuration': {'embedding_dim': 8, 'hidden_dim': 16, 'window_size': 5,
                                'learning_rate': learning_rate, 'batch_size': batch_size,
                                'context_dropout': context_dropout, 'max_epochs': max_epochs,
                                'patience': patience, 'selection': 'minimum validation loss'},
              'labels': list(LABELS),
              'baselines': {
                  'constant_label_accuracy': {
                      label: sum(row['label'] == label for row in test) / len(test)
                      for label in LABELS},
                  'target_only_expected_accuracy':
                      sum(max(counts.values()) for counts in (
                          Counter(row['label'] for row in test if row['target'] == target)
                          for target in {row['target'] for row in test})) / len(test),
                  'note': 'Original ambiguous targets have balanced noun/verb labels. Additional noun-only and other-only targets can be distinguished by identity.'}}

    predictions = []
    for name, rows in [('train', training), ('validation', validation), ('test', test)]:
        report[name], examples = evaluate(model, rows)
        predictions.extend(examples)
    # 5. Save evidence for the walkthrough: final metrics, epoch-by-epoch
    # progress, and individual predictions. best_model.npz includes vocabulary
    # and learned parameters; inference loads it without training again.
    (output_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    (output_dir / 'history.json').write_text(json.dumps(history, indent=2) + '\n')
    (output_dir / 'predictions.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in predictions))
    if verbose:
        print(json.dumps({key: report[key] for key in ['parameter_count', 'training_seconds', 'best_epoch', 'epochs_run']}, indent=2))
        for split in ['train', 'validation', 'test']:
            print(f"{split}: accuracy={report[split]['accuracy']:.3f} loss={report[split]['loss']:.4f}")
    if callback is not None:
        callback({'type': 'complete', 'report': report})
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path(__file__).parent)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).parent / 'results')
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()
    train(args.data_dir, args.output_dir, args.seed)
