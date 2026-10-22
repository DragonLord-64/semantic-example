# Student exercise answer key

## TODOs 1 and 2

Replace the TODO sections in `student_exercise.py` with:

```python
def main():
    predictor = Predictor.from_checkpoint()

    for sentence in SENTENCES:
        result = predictor.predict_word(sentence, TARGET)
        print(f"{sentence} {result['prediction']} "
              f"(verb: {result['verb_probability']:.1%})")
```

## Labels and comparison

Model results below use the current saved checkpoint. Retraining may change them.

| Sentence | Correct label | Model label | P(verb) |
| --- | --- | --- | ---: |
| They watch the birds. | verb | verb | 100.0% |
| The watch is broken. | noun | noun | 0.1% |
| We watch the game. | verb | verb | 100.0% |
| She bought a watch. | noun | noun | 0.0% |
| Please watch the dog. | verb | verb | 100.0% |
| His watch looks expensive. | noun | noun | 29.4% |
| I watch. | verb | noun | 0.5% |

“I watch.” is the surprise: “watch” is an action, but the model predicts noun with about 99.5% probability. Confidence does not guarantee correctness; this short context differs from the training patterns.

## Two new sentences

- “We watch the stars.” — verb: an action.
- “My watch is new.” — noun: an object.

## What changes?

The surrounding words and sentence boundaries change. Their embeddings occupy different positions in the model's five-token window, changing its prediction even though “watch” stays the same.

## Bonus

Try `TARGET = "play"` with “They play the game.” (verb) and “The play is funny.” (noun). Unknown words share the same `<UNK>` vector, so the model cannot distinguish them by identity.
