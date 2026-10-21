"""Train on train, select by validation loss, then evaluate test once."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time
import numpy as np
from model import Adam, ContextModel, UNK, PAD, build_vocab, encode, sigmoid


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def evaluate(model, rows):
    windows, labels = encode(rows, model.vocab)
    logits = model.forward(windows)[0]
    probabilities = sigmoid(logits)
    predicted = probabilities >= 0.5
    correct = predicted == labels
    confusion = np.zeros((2, 2), dtype=int)
    np.add.at(confusion, (labels.astype(int), predicted.astype(int)), 1)
    context_mask = np.ones_like(windows, dtype=bool)
    context_mask[:, 2] = False
    context_mask &= windows != model.vocab[PAD]
    oov = (windows == model.vocab[UNK]) & context_mask
    groups = {}
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
        'loss': float(np.mean(np.logaddexp(0, logits) - labels * logits)),
        'accuracy': float(correct.mean()),
        'confusion_rows_actual_columns_predicted_noun_verb': confusion.tolist(),
        'context_oov_rate': float(oov.sum() / max(1, context_mask.sum())),
        'examples_with_oov_context': int(oov.any(axis=1).sum()),
        **groups,
    }, [dict(row, verb_probability=float(probabilities[i]),
             prediction='verb' if predicted[i] else 'noun', correct=bool(correct[i]),
             window_ids=windows[i].tolist()) for i, row in enumerate(rows)]


def train(data_dir, output_dir, seed=7, max_epochs=300, patience=30,
          learning_rate=0.02, batch_size=32, context_dropout=0.1,
          callback=None, verbose=True):
    output_dir.mkdir(parents=True, exist_ok=True)
    training = read_rows(data_dir / 'train.jsonl')
    validation = read_rows(data_dir / 'validation.jsonl')
    vocab = build_vocab(training)
    model = ContextModel(vocab, seed=seed)
    windows, labels = encode(training, vocab)
    optimizer = Adam(model.params, learning_rate)
    rng = np.random.default_rng(seed + 1)
    initial, _ = evaluate(model, training)
    initial_validation, _ = evaluate(model, validation)
    history = [dict(epoch=0, train_loss=initial['loss'], train_accuracy=initial['accuracy'],
                    validation_loss=initial_validation['loss'],
                    validation_accuracy=initial_validation['accuracy'])]
    # A concrete, plain SGD update trace, independent of the Adam training run.
    one_x, one_y = windows[:1], labels[:1]
    before = float(model.predict(one_x)[0])
    loss, gradients = model.loss_and_gradients(one_x, one_y)
    original = {key: value.copy() for key, value in model.params.items()}
    for key in model.params:
        model.params[key] -= 0.1 * gradients[key]
    after = float(model.predict(one_x)[0])
    trace = {'example': training[0], 'window_ids': one_x[0].tolist(),
             'verb_probability_before': before, 'loss_before': loss,
             'verb_probability_after': after,
             'loss_after': model.loss_and_gradients(one_x, one_y)[0],
             'optimizer': 'one illustrative SGD step; full run uses Adam',
             'learning_rate': 0.1,
             'parameters_before': {k: v.tolist() for k, v in original.items()},
             'gradients': {k: v.tolist() for k, v in gradients.items()},
             'parameters_after': {k: v.tolist() for k, v in model.params.items()}}
    model.params = original
    (output_dir / 'one_update.json').write_text(json.dumps(trace, indent=2) + '\n')
    best_loss, best_epoch, stale = float('inf'), 0, 0
    callback_seconds = 0.
    started = time.perf_counter()

    def emit_epoch(metrics):
        nonlocal callback_seconds
        if callback is not None:
            event = dict(history[-1], type='epoch',
                         validation_confusion=metrics['confusion_rows_actual_columns_predicted_noun_verb'],
                         learning_rate=learning_rate, best_epoch=best_epoch,
                         compute_seconds=time.perf_counter() - started - callback_seconds)
            callback_started = time.perf_counter()
            callback(event)
            callback_seconds += time.perf_counter() - callback_started

    emit_epoch(initial_validation)
    for epoch in range(1, max_epochs + 1):
        for indices in np.array_split(rng.permutation(len(labels)),
                                      int(np.ceil(len(labels) / batch_size))):
            batch = windows[indices].copy()
            mask = rng.random(batch.shape) < context_dropout
            mask[:, 2] = False
            mask &= batch != vocab[PAD]
            batch[mask] = vocab[UNK]
            _, gradients = model.loss_and_gradients(batch, labels[indices])
            optimizer.update(model.params, gradients)
        train_metrics, _ = evaluate(model, training)
        val_metrics, _ = evaluate(model, validation)
        history.append(dict(epoch=epoch, train_loss=train_metrics['loss'],
                            train_accuracy=train_metrics['accuracy'],
                            validation_loss=val_metrics['loss'],
                            validation_accuracy=val_metrics['accuracy']))
        if val_metrics['loss'] < best_loss - 1e-6:
            best_loss, best_epoch, stale = val_metrics['loss'], epoch, 0
            model.save(output_dir / 'best_model.npz')
        else:
            stale += 1
        emit_epoch(val_metrics)
        if stale >= patience:
            break
    elapsed = time.perf_counter() - started - callback_seconds
    model = ContextModel.load(output_dir / 'best_model.npz')
    # Only open test after selection is complete. No selection uses these results.
    test = read_rows(data_dir / 'test.jsonl')
    report = {'seed': seed, 'vocabulary_size': len(vocab),
              'parameter_count': sum(p.size for p in model.params.values()),
              'training_seconds': elapsed, 'epochs_run': epoch, 'best_epoch': best_epoch,
              'configuration': {'embedding_dim': 8, 'hidden_dim': 16, 'window_size': 5,
                                'learning_rate': learning_rate, 'batch_size': batch_size,
                                'context_dropout': context_dropout, 'max_epochs': max_epochs,
                                'patience': patience, 'selection': 'minimum validation loss'},
              'baselines': {'constant_noun_accuracy': sum(r['label'] == 'noun' for r in test) / len(test),
                            'target_only_expected_accuracy': 0.5,
                            'note': 'Every target has balanced labels in each split; target identity alone cannot distinguish labels.'}}
    predictions = []
    for name, rows in [('train', training), ('validation', validation), ('test', test)]:
        report[name], examples = evaluate(model, rows)
        predictions.extend(examples)
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
