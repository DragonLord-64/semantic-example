"""Predictor API checks and a regression for the shipped example checkpoint."""
from contextlib import redirect_stdout
from io import StringIO
import unittest

import numpy as np

from inference import Predictor
from model import ContextModel, LABELS, PAD, UNK
from predict import classify, print_prediction


class FixedModel:
    def __init__(self, probabilities):
        self.vocab = {PAD: 0, UNK: 1, 'they': 2, 'watch': 3, 'the': 4, 'birds': 5}
        self.targets = frozenset(self.vocab) - {PAD, UNK}
        self.probabilities = np.array(probabilities)

    def predict(self, windows):
        return np.tile(self.probabilities, (len(windows), 1))


class PredictorTests(unittest.TestCase):
    def test_other_is_a_model_output_in_the_predictor(self):
        model = ContextModel(FixedModel([0, 0, 1]).vocab)
        model.params['W2'].fill(0)
        model.params['b2'][:] = [-3, -2, 4]
        result = Predictor(model).predict('They watch.', 0)
        self.assertEqual(result['prediction'], 'other')
        self.assertGreater(result['other_probability'], 0.99)
        self.assertAlmostEqual(sum(result[f'{label}_probability'] for label in LABELS), 1)

    def test_reports_each_probability_and_selects_argmax(self):
        for winner, label in enumerate(LABELS):
            with self.subTest(label=label):
                probabilities = [0.2, 0.2, 0.2]
                probabilities[winner] = 0.6
                result = Predictor(FixedModel(probabilities)).predict('They watch.', 1)
                self.assertEqual(result['prediction'], label)
                self.assertEqual(result['status'], 'classified')
                for index, name in enumerate(LABELS):
                    self.assertAlmostEqual(result[f'{name}_probability'], probabilities[index])

    def test_sentence_prediction_classifies_every_token_including_other(self):
        predictor = Predictor(FixedModel([0.1, 0.2, 0.7]))
        results = predictor.predict_sentence('They watch the birds.')
        self.assertEqual([result['token'] for result in results],
                         ['they', 'watch', 'the', 'birds'])
        self.assertEqual([result['target_index'] for result in results], [0, 1, 2, 3])
        self.assertTrue(all(result['prediction'] == 'other' for result in results))
        self.assertTrue(all(result['status'] == 'classified' for result in results))

    def test_interactive_prediction_reports_other_probability(self):
        model = FixedModel([0.1, 0.2, 0.7])
        result = classify(model, 'They watch.', 0)
        self.assertEqual(result['prediction'], 'other')
        self.assertEqual(result['target'], 'they')
        output = StringIO()
        with redirect_stdout(output):
            print_prediction(result)
        self.assertEqual(output.getvalue().strip(), 'they: other (70.0%)')

    def test_repeated_word_uses_selected_occurrence_context(self):
        predictor = Predictor(FixedModel([0.1, 0.8, 0.1]))
        sentence = 'They watch the watch.'
        first = predictor.predict(sentence, 1)
        second = predictor.predict(sentence, 3)
        self.assertEqual(first['context'], [PAD, 'they', 'watch', 'the', 'watch'])
        self.assertEqual(second['context'], ['watch', 'the', 'watch', PAD, PAD])
        self.assertNotEqual(first['window_ids'], second['window_ids'])
        with self.assertRaises(ValueError):
            predictor.predict_word(sentence, 'watch')

    def test_invalid_index_is_rejected(self):
        predictor = Predictor(FixedModel([0.1, 0.2, 0.7]))
        for index in [-1, 2, '1']:
            with self.subTest(index=index), self.assertRaises(ValueError):
                predictor.predict('They watch.', index)

    def test_empty_sentence_has_no_predictions(self):
        predictor = Predictor(FixedModel([0.1, 0.2, 0.7]))
        self.assertEqual(predictor.predict_sentence('...'), [])

    def test_checkpoint_finds_noun_verb_and_other_in_example(self):
        results = Predictor.from_checkpoint().predict_sentence('They watch the birds.')
        self.assertEqual([result['prediction'] for result in results],
                         ['other', 'verb', 'other', 'noun'])


if __name__ == '__main__':
    unittest.main()
