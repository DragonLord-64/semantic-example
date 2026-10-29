"""student-example: how to use the prediction module.

Run: python student_example.py
The saved model is loaded once and reused for every sentence.

predictor.predict_word(sentence, TARGET) returns a dictionary containing
'prediction' (noun, verb, or other) and each label's probability (0 to 1).
For repeated words, use predictor.predict(sentence, target_index) instead;
target_index is a zero-based position in model.tokenize(sentence).
"""
if __package__:
    from .inference import Predictor
else:
    from inference import Predictor

TARGET = "watch"
SENTENCES = [
    "They watch the birds.",
    "The watch is broken.",
    "We watch the game.",
    "She bought a watch.",
    "Please watch the dog.",
    "His watch looks expensive.",
    "I watch.",  # Does a confident prediction have to be correct?
]


def main():
    predictor = Predictor.from_checkpoint()

    for sentence in SENTENCES:
        result = predictor.predict_word(sentence, TARGET)
        print(sentence, result["prediction"], result["verb_probability"])
        # print(sentence, result["prediction"], result["noun_probability"])
        # print(sentence, result["prediction"], result["other_probability"])

    # Compare the predictions with your own labels, then add two sentences.
    # What changes between sentences if TARGET stays the same?
    # Try TARGET = "the" in "They watch the birds." to see the other label.
    # Unseen words share an unknown vector; confidence can be misleading.


if __name__ == "__main__":
    main()
