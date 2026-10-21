"""CLI and importable inference using saved weights; no training on import."""
import argparse
from pathlib import Path
import sys

from model import ContextModel, PAD, UNK, encode, tokenize

HERE = Path(__file__).resolve().parent
CHECKPOINT = HERE / 'results' / 'best_model.npz'
# Supervised targets, not every word in the context vocabulary.
TARGETS = frozenset(('watch', 'play', 'call', 'walk', 'run', 'drink', 'kick',
                     'park', 'plant', 'dance', 'brush', 'wave', 'smile',
                     'answer', 'dream', 'light'))


def load_model(checkpoint=CHECKPOINT):
    """Load saved weights without retraining or modifying the checkpoint."""
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f'No saved checkpoint found: {path}. '
                                f'Run training first: python {HERE / "run_training.py"}')
    return ContextModel.load(path)


class Predictor:
    def __init__(self, model):
        self.model = model

    @classmethod
    def from_checkpoint(cls, checkpoint=CHECKPOINT):
        return cls(load_model(checkpoint))

    def predict(self, sentence, target_index):
        """Classify one selected occurrence at a zero-based tokenized index."""
        words = tokenize(sentence)
        if not isinstance(target_index, int) or not 0 <= target_index < len(words):
            raise ValueError('Target index must be a valid zero-based token position.')
        target = words[target_index]
        row = {'id': 'inference', 'sentence': sentence, 'target': target,
               'target_index': target_index, 'label': 'noun'}
        windows, _ = encode([row], self.model.vocab)
        verb_probability = float(self.model.predict(windows)[0])
        context = [words[i] if 0 <= i < len(words) else PAD
                   for i in range(target_index - 2, target_index + 3)]
        return {'token': target, 'target_index': target_index,
                'trained_target': target in TARGETS,
                'status': 'classified', 'context': context,
                'window_ids': windows[0].tolist(),
                'unknown_context': sorted({word for offset, word in enumerate(context)
                                           if offset != 2 and word != PAD
                                           and word not in self.model.vocab}),
                'unknown_target': target not in self.model.vocab,
                'noun_probability': 1 - verb_probability,
                'verb_probability': verb_probability,
                'prediction': 'verb' if verb_probability >= 0.5 else 'noun'}

    def predict_word(self, sentence, word):
        """Select a word occurring exactly once; repeated words need an index."""
        words = tokenize(sentence)
        positions = [i for i, token in enumerate(words) if token == word.strip().lower()]
        if not positions:
            raise ValueError(f'{word!r} is not in the tokenized sentence.')
        if len(positions) > 1:
            raise ValueError(f'{word!r} occurs more than once; select a zero-based '
                             f'token index: {positions}.')
        return self.predict(sentence, positions[0])

    def predict_sentence(self, sentence):
        """Optional demo helper: classify known targets only, not all-word tagging."""
        return [self.predict(sentence, i) if word in TARGETS else
                {'token': word, 'target_index': i, 'status': 'unsupported'}
                for i, word in enumerate(tokenize(sentence))]


def predict_sentence(sentence, predictor=None):
    """Pass an existing Predictor to reuse already-loaded weights."""
    if predictor is None:
        predictor = Predictor.from_checkpoint()
    return predictor.predict_sentence(sentence)


def display(sentence, records):
    print(f'Sentence: {sentence}')
    print('Positions below are 1-based; Python API indices are 0-based.')
    print(f"{'#':>3}  {'Token':<16} {'Result':<14} {'P(noun)':>9} {'P(verb)':>9}")
    print('-' * 59)
    classified = []
    for result in records:
        if result['status'] == 'classified':
            classified.append(result)
            print(f"{result['target_index'] + 1:>3}  {result['token']:<16} "
                  f"{result['prediction']:<14} {result['noun_probability']:>9.2%} "
                  f"{result['verb_probability']:>9.2%}")
        else:
            print(f"{result['target_index'] + 1:>3}  {result['token']:<16} "
                  f"{'not classified':<14} {'-':>9} {'-':>9}")
    if not classified:
        print('\nNo supported target words found. Try: They watch the watch.')
    for result in classified:
        context = ' | '.join(f'[{word}]' if offset == 2 else word
                             for offset, word in enumerate(result['context']))
        print(f"\nToken {result['target_index'] + 1} context: {context}")
        if result['unknown_target']:
            print(f'Warning: target absent from checkpoint vocabulary; mapped to {UNK}.')
        if result['unknown_context']:
            print(f'Unseen context mapped to {UNK}: ' + ', '.join(result['unknown_context']))
        if not result['trained_target']:
            print('Warning: this word was not a supervised target during training.')
    print('\nThis binary model classifies the selected occurrence as noun or verb.')
    print('Other roles are unsupported; probabilities do not guarantee correctness.')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('target', help='Target word, or a zero-based token index.')
    parser.add_argument('sentence', help='One quoted sentence; no input file needed.')
    args = parser.parse_args(argv)
    if not tokenize(args.sentence):
        parser.error('The sentence must contain alphabetic words.')
    try:
        predictor = Predictor.from_checkpoint()
        result = (predictor.predict(args.sentence, int(args.target))
                  if args.target.lstrip('-').isdecimal() else
                  predictor.predict_word(args.sentence, args.target))
    except (OSError, ValueError, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    display(args.sentence, [result])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
