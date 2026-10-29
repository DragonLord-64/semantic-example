"""Check the reference solution's word-by-word use of the Predictor API."""
import unittest

from noun_verb.student_exercise_answers import find_nouns_and_verbs


class RecordingPredictor:
    def __init__(self, labels):
        self.labels = labels
        self.calls = []

    def predict(self, sentence, target_index):
        self.calls.append((sentence, target_index))
        return {'prediction': self.labels[target_index]}


class SentenceExerciseTests(unittest.TestCase):
    def test_preserves_sentence_context_and_repeated_word_indices(self):
        sentence = 'They watch the watch.'
        predictor = RecordingPredictor(['other', 'verb', 'other', 'noun'])
        self.assertEqual(find_nouns_and_verbs(sentence, predictor),
                         {'nouns': ['watch'], 'verbs': ['watch']})
        self.assertEqual(predictor.calls, [(sentence, index) for index in range(4)])

    def test_filters_other_and_keeps_repeated_results(self):
        sentence = 'They watch the birds, and watch the birds!'
        labels = ['other', 'verb', 'other', 'noun', 'other', 'verb', 'other', 'noun']
        predictor = RecordingPredictor(labels)
        self.assertEqual(find_nouns_and_verbs(sentence, predictor),
                         {'nouns': ['birds', 'birds'], 'verbs': ['watch', 'watch']})
        self.assertEqual(predictor.calls, [(sentence, index) for index in range(8)])

    def test_empty_input_returns_empty_lists_without_predictions(self):
        predictor = RecordingPredictor([])
        self.assertEqual(find_nouns_and_verbs('...', predictor), {'nouns': [], 'verbs': []})
        self.assertEqual(predictor.calls, [])


if __name__ == '__main__':
    unittest.main()
