"""Reference solution: python noun_verb/student_exercise_answers.py. Uses saved weights."""
from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference import Predictor
from model import tokenize
from noun_verb.student_exercise import SENTENCE


def find_nouns_and_verbs(sentence, predictor):
    """Return predicted nouns and verbs, preserving order and repeated words."""
    words_by_label = {"nouns": [], "verbs": []}
    groups = {"noun": "nouns", "verb": "verbs"}
    for index, word in enumerate(tokenize(sentence)):
        # Keep the whole sentence as context; select each token by its index.
        # Tokenization handles punctuation, and indices handle repeated words.
        result = predictor.predict(sentence, target_index=index)
        group = groups.get(result["prediction"])
        if group is not None:
            words_by_label[group].append(word)
    return words_by_label


def main():
    predictor = Predictor.from_checkpoint()
    words_by_label = find_nouns_and_verbs(SENTENCE, predictor)
    print(SENTENCE)
    print("Nouns:", words_by_label["nouns"])
    print("Verbs:", words_by_label["verbs"])


if __name__ == "__main__":
    main()
