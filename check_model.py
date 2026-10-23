"""Numerical checks independent of the held-out test data."""
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from model import ContextModel, PAD, UNK, softmax


def main():
    vocab = {PAD: 0, UNK: 1, 'the': 2, 'watch': 3, 'they': 4}
    model = ContextModel(vocab)
    # Repeated 'the' and repeated PAD deliberately exercise scatter-add gradients.
    windows = np.array([[0, 2, 3, 2, 0], [0, 4, 3, 0, 0], [0, 0, 2, 3, 0]])
    labels = np.array([0, 1, 2], dtype=np.int64)
    loss, gradients = model.loss_and_gradients(windows, labels)
    eps, worst = 1e-5, 0.
    for key, param in model.params.items():
        error = 0.
        for index in np.ndindex(param.shape):
            original = param[index]
            param[index] = original + eps
            plus = model.loss_and_gradients(windows, labels)[0]
            param[index] = original - eps
            minus = model.loss_and_gradients(windows, labels)[0]
            param[index] = original
            numeric = (plus - minus) / (2 * eps)
            error = max(error, abs(numeric - gradients[key][index]))
        print(f'{key}: max absolute gradient error {error:.3g}')
        assert error < 1e-7, (key, error)
        worst = max(worst, error)
    for key in model.params:
        model.params[key] -= 0.1 * gradients[key]
    assert model.loss_and_gradients(windows, labels)[0] < loss
    before = model.predict(windows)
    with TemporaryDirectory() as directory:
        checkpoint = Path(directory) / 'model.npz'
        model.save(checkpoint)
        restored = ContextModel.load(checkpoint)
        np.testing.assert_array_equal(before, restored.predict(windows))
    np.testing.assert_allclose(softmax(np.array([[-1000., 0., 1000.]])), [[0., 0., 1.]])
    # Extreme logits must also produce finite losses and gradients.
    for extreme in [-1000., 1000.]:
        model.params['b2'][0] = extreme
        value, grads = model.loss_and_gradients(windows, labels)
        assert np.isfinite(value)
        assert all(np.isfinite(g).all() for g in grads.values())
    print(f'PASS: all parameter entries checked; worst error={worst:.3g}; loss decreases; serialization exact; extreme logits finite.')


if __name__ == '__main__':
    main()
