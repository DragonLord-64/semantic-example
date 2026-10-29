"""Example of importing inference and loading weights from another script."""
if __package__:
    from .inference import Predictor
else:
    from inference import Predictor


def main():
    predictor = Predictor.from_checkpoint()  # Load once; never trains.
    sentence = 'They watch the birds.'
    result = predictor.predict_word(sentence, 'watch')
    print(f"{result['token']}: {result['prediction']} "
          f"(P(verb)={result['verb_probability']:.4f})")
    # Select a specific occurrence using a zero-based tokenizer index.
    print('Selected occurrence:', predictor.predict(sentence, target_index=1))
    # Other parts of speech receive the "other" label.
    result = predictor.predict_word(sentence, 'the')
    print(f"{result['token']}: {result['prediction']} "
          f"(P(other)={result['other_probability']:.4f})")


if __name__ == '__main__':
    main()
