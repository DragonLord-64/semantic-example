"""Exercise: classify the same word in different sentences.

Run: python student_exercise.py
Finish the TODOs, then run again. Use the saved model; do not retrain it.
"""
from inference import Predictor

TARGET = "watch"
SENTENCES = [
    "They watch the birds.",
    "The watch is broken.",
    "We watch the game.",
    "She bought a watch.",
    "Please watch the dog.",
    "His watch looks expensive.",
    "I watch.",  # Challenge: does a confident prediction have to be correct?
]


def main():
    # 1. Before running the model, predict noun/verb yourself for each sentence.

    # TODO 1: Load the saved model.
    predictor = None

    if predictor is None:
        print("Finish TODO 1 and TODO 2 in student_exercise.py, then run again.")
        return

    # TODO 2: Print each sentence, TARGET's predicted label, and verb probability.
    pass

    # 3. Compare with your own labels. Which predictions surprised you?
    # 4. Add two sentences of your own, with one occurrence of TARGET each.
    # 5. Explain: what changes between sentences if TARGET stays the same?
    # Bonus: try another noun/verb word. Unseen words use an unknown vector.


if __name__ == "__main__":
    main()
