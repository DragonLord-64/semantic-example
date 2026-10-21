"""Interactively classify one selected word using the saved best checkpoint."""
from pathlib import Path
import sys

try:
    from model import ContextModel, PAD, UNK, encode, tokenize
except ModuleNotFoundError as exc:
    if exc.name != 'numpy':
        raise
    print('NumPy is required. Install requirements.txt into your Python environment.',
          file=sys.stderr)
    raise SystemExit(1)


HERE = Path(__file__).resolve().parent
CHECKPOINT = HERE / 'results' / 'best_model.npz'


def select_target(words, choice):
    """Return a zero-based position; numbered CLI choices are one-based."""
    choice = choice.strip().lower()
    if choice.isdecimal():
        index = int(choice) - 1
        if 0 <= index < len(words):
            return index
        raise ValueError(f'Choose a token number from 1 to {len(words)}.')
    matches = [i for i, word in enumerate(words) if word == choice]
    if not matches:
        raise ValueError('That word is not in the displayed tokens.')
    if len(matches) > 1:
        numbers = ', '.join(str(i + 1) for i in matches)
        raise ValueError(f'That word occurs more than once; choose token {numbers}.')
    return matches[0]


def classify(model, sentence, index):
    """Use precisely the same tokenizer and context encoder as training."""
    words = tokenize(sentence)
    if not 0 <= index < len(words):
        raise ValueError('Target index is outside the sentence.')
    row = {'id': 'interactive', 'sentence': sentence, 'target': words[index],
           'target_index': index, 'label': 'noun'}
    windows, _ = encode([row], model.vocab)
    verb_probability = float(model.predict(windows)[0])
    context = [words[i] if 0 <= i < len(words) else PAD
               for i in range(index - 2, index + 3)]
    return {'target': words[index], 'target_index': index,
            'context': context, 'window_ids': windows[0].tolist(),
            'unknown_context': sorted({word for offset, word in enumerate(context)
                                       if offset != 2 and word != PAD
                                       and word not in model.vocab}),
            'unknown_target': words[index] not in model.vocab,
            'noun_probability': 1 - verb_probability,
            'verb_probability': verb_probability,
            'prediction': 'verb' if verb_probability >= 0.5 else 'noun'}


def print_prediction(result):
    display = [f'[{word}]' if i == 2 else word
               for i, word in enumerate(result['context'])]
    print('\nContext (two words on each side): ' + ' | '.join(display))
    print(f"  noun  {result['noun_probability']:7.2%}")
    print(f"  verb  {result['verb_probability']:7.2%}")
    print(f"Prediction for token {result['target_index'] + 1} "
          f"({result['target']}): {result['prediction']}")
    if result['unknown_target']:
        print(f"Warning: target was unseen in training and mapped to {UNK}.")
    if result['unknown_context']:
        print(f"Unseen context mapped to {UNK}: " + ', '.join(result['unknown_context']))
    print('These are model probabilities, not a guarantee of correctness.\n')


def main(checkpoint=CHECKPOINT):
    if not checkpoint.is_file():
        print(f'No saved checkpoint found: {checkpoint}', file=sys.stderr)
        print(f'Run training first: python3 {HERE / "run_training.py"}', file=sys.stderr)
        return 1
    try:
        model = ContextModel.load(checkpoint)
    except (OSError, ValueError, KeyError) as exc:
        print(f'Could not load checkpoint: {exc}', file=sys.stderr)
        return 1
    print(f'Loaded saved weights: {checkpoint}')
    print('Classify ONE selected word as noun or verb; other roles are unsupported.')
    print('Try: The watch is broken.  Or: They watch the birds.')
    print('Type :quit or submit an empty sentence to finish.')
    try:
        while True:
            sentence = input('\nSentence> ').strip()
            if not sentence or sentence.lower() == ':quit':
                break
            words = tokenize(sentence)
            if not words:
                print('Enter a sentence containing alphabetic words.')
                continue
            print('Tokens: ' + '  '.join(f'{i + 1}:{word}' for i, word in enumerate(words)))
            while True:
                choice = input('Target word or token number (:back for another sentence)> ')
                if choice.strip().lower() == ':quit':
                    return 0
                if choice.strip().lower() == ':back':
                    break
                try:
                    index = select_target(words, choice)
                except ValueError as exc:
                    print(exc)
                    continue
                print_prediction(classify(model, sentence, index))
                break
    except (EOFError, KeyboardInterrupt):
        print('\nFinished.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
