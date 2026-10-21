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

    # TODO 1: Replace None by loading the saved weights ONCE.
    # Hint: Predictor.from_checkpoint() returns an object you can reuse.
    predictor = None

    if predictor is None:
        print("Finish TODO 1 and TODO 2 in student_exercise.py, then run again.")
        return

    # TODO 2: Loop over SENTENCES. For each sentence:
    # - Call predictor.predict_word(sentence, TARGET).
    # - The returned dictionary has 'prediction' and 'verb_probability' keys.
    # - Print the sentence, predicted label, and probability of verb.
    # Hint: f"{probability:.1%}" formats a number as a percentage.
    pass

    # 3. Compare with your own labels. Which predictions surprised you?
    # 4. Add two sentences of your own, with one occurrence of TARGET each.
    # 5. Explain: what changes between sentences if TARGET stays the same?
    # Bonus: try another noun/verb word. Unseen words use an unknown vector.


if __name__ == "__main__":
    main()
