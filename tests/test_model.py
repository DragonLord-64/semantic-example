"""Three-label numerical and serialization checks, without training a model."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from model import ContextModel, LABELS, PAD, UNK, encode


class ThreeLabelModelTests(unittest.TestCase):
    def setUp(self):
        self.vocab = {PAD: 0, UNK: 1, 'they': 2, 'watch': 3, 'the': 4}
        self.model = ContextModel(self.vocab)
        self.windows = np.array([[0, 2, 3, 4, 3], [3, 4, 3, 0, 0]])

    def test_encode_uses_integer_indices_for_all_three_labels(self):
        rows = [dict(id=label, sentence='They watch the watch.', target='watch',
                     target_index=1, label=label) for label in LABELS]
        windows, labels = encode(rows, self.vocab)
        self.assertEqual(tuple(LABELS), ('noun', 'verb', 'other'))
        self.assertTrue(np.issubdtype(labels.dtype, np.integer))
        np.testing.assert_array_equal(labels, [0, 1, 2])
        np.testing.assert_array_equal(windows, [self.windows[0]] * 3)

    def test_encode_rejects_unknown_labels(self):
        row = dict(id='bad-label', sentence='They watch.', target='watch',
                   target_index=1, label='adjective')
        with self.assertRaises(ValueError):
            encode([row], self.vocab)

    def test_encode_selects_repeated_word_by_index(self):
        rows = [dict(id=str(index), sentence='They watch the watch.', target='watch',
                     target_index=index, label=label)
                for index, label in [(1, 'verb'), (3, 'noun')]]
        windows, labels = encode(rows, self.vocab)
        np.testing.assert_array_equal(windows, self.windows)
        np.testing.assert_array_equal(labels, [1, 0])

    def test_probabilities_have_three_normalized_columns(self):
        probabilities = self.model.predict(self.windows)
        self.assertEqual(probabilities.shape, (2, 3))
        self.assertTrue(np.isfinite(probabilities).all())
        self.assertTrue(((probabilities >= 0) & (probabilities <= 1)).all())
        np.testing.assert_allclose(probabilities.sum(axis=1), [1, 1])

    def test_extreme_logits_keep_probabilities_loss_and_gradients_finite(self):
        self.model.params['W2'].fill(0)
        self.model.params['b2'][:] = [-1000, 0, 1000]
        np.testing.assert_allclose(self.model.predict(self.windows),
                                   [[0, 0, 1], [0, 0, 1]])
        loss, gradients = self.model.loss_and_gradients(
            self.windows, np.array([0, 2], dtype=np.int64))
        self.assertTrue(np.isfinite(loss))
        self.assertTrue(all(np.isfinite(value).all() for value in gradients.values()))

    def test_other_class_gradients_match_finite_differences(self):
        labels = np.array([2, 2], dtype=np.int64)
        loss, gradients = self.model.loss_and_gradients(self.windows, labels)
        # Include the new output column and an embedding shared by repeated words.
        entries = [('W2', (0, 2)), ('b2', (2,)), ('W1', (0, 0)),
                   ('b1', (0,)), ('embedding', (3, 0))]
        epsilon = 1e-5
        for name, index in entries:
            with self.subTest(parameter=name):
                parameter = self.model.params[name]
                original = parameter[index]
                try:
                    parameter[index] = original + epsilon
                    plus = self.model.loss_and_gradients(self.windows, labels)[0]
                    parameter[index] = original - epsilon
                    minus = self.model.loss_and_gradients(self.windows, labels)[0]
                finally:
                    parameter[index] = original
                numeric = (plus - minus) / (2 * epsilon)
                self.assertAlmostEqual(numeric, gradients[name][index], places=7)
        for name in self.model.params:
            self.model.params[name] -= 0.01 * gradients[name]
        self.assertLess(self.model.loss_and_gradients(self.windows, labels)[0], loss)

    def test_save_load_preserves_all_three_probabilities(self):
        before = self.model.predict(self.windows)
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'model.npz'
            self.model.save(checkpoint)
            restored = ContextModel.load(checkpoint)
        self.assertEqual(restored.vocab, self.vocab)
        np.testing.assert_array_equal(restored.predict(self.windows), before)


if __name__ == '__main__':
    unittest.main()
