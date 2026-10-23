"""Interactively classify one selected word using the saved best checkpoint."""
from pathlib import Path
import sys

try:
    from model import ContextModel, tokenize
    from inference import Predictor
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
    result = Predictor(model).predict(sentence, index)
    return dict(result, target=result['token'])


def print_prediction(result):
    label = result['prediction']
    print(f"{result['target']}: {label} ({result[label + '_probability']:.1%})")


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
    print('Noun, verb, or other. Empty sentence or :quit to exit; :back to change sentence.')
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
                choice = input('Word or number> ')
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
        print()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
