"""A small context classifier, with explicit NumPy forward/backward calculations."""
import json
import re
import numpy as np

TOKEN_PATTERN = r"[A-Za-z]+(?:'[A-Za-z]+)?"
PAD, UNK = "<PAD>", "<UNK>"


def tokenize(sentence):
    return re.findall(TOKEN_PATTERN, sentence.lower())


def build_vocab(rows):
    words = sorted({word for row in rows for word in tokenize(row['sentence'])})
    return {word: i for i, word in enumerate([PAD, UNK] + words)}


def encode(rows, vocab):
    windows = []
    for row in rows:
        words = tokenize(row['sentence'])
        pos = row['target_index']
        if not 0 <= pos < len(words) or words[pos] != row['target'].lower():
            raise ValueError(f"Invalid target in {row['id']}")
        windows.append([vocab.get(words[i], vocab[UNK]) if 0 <= i < len(words)
                        else vocab[PAD] for i in range(pos - 2, pos + 3)])
    return np.array(windows, dtype=np.int64), np.array(
        [row['label'] == 'verb' for row in rows], dtype=np.float64)


def sigmoid(logits):
    # exp(-logaddexp(0, -z)) is stable even for extreme logits.
    return np.exp(-np.logaddexp(0, -logits))


class ContextModel:
    def __init__(self, vocab, seed=7, embedding_dim=8, hidden_dim=16):
        self.vocab = vocab
        rng = np.random.default_rng(seed)
        self.params = {
            'embedding': rng.normal(0, 0.1, (len(vocab), embedding_dim)),
            'W1': rng.normal(0, 1 / np.sqrt(5 * embedding_dim),
                             (5 * embedding_dim, hidden_dim)),
            'b1': np.zeros(hidden_dim),
            'W2': rng.normal(0, 1 / np.sqrt(hidden_dim), hidden_dim),
            'b2': np.zeros(1),
        }

    def forward(self, windows):
        flat = self.params['embedding'][windows].reshape(len(windows), -1)
        hidden = np.tanh(flat @ self.params['W1'] + self.params['b1'])
        logits = hidden @ self.params['W2'] + self.params['b2'][0]
        return logits, (flat, hidden)

    def loss_and_gradients(self, windows, labels):
        logits, (flat, hidden) = self.forward(windows)
        loss = np.mean(np.logaddexp(0, logits) - labels * logits)
        dlogits = (sigmoid(logits) - labels) / len(labels)
        dhidden = dlogits[:, None] * self.params['W2'] * (1 - hidden ** 2)
        dflat = dhidden @ self.params['W1'].T
        gradients = {
            'embedding': np.zeros_like(self.params['embedding']),
            'W1': flat.T @ dhidden,
            'b1': dhidden.sum(axis=0),
            'W2': hidden.T @ dlogits,
            'b2': np.array([dlogits.sum()]),
        }
        # Accumulate repeated token IDs; indexed += would drop repeated entries.
        np.add.at(gradients['embedding'], windows,
                  dflat.reshape(len(windows), 5, -1))
        return float(loss), gradients

    def predict(self, windows):
        return sigmoid(self.forward(windows)[0])

    def save(self, path):
        np.savez(path, vocab=np.array(json.dumps(self.vocab)), **self.params)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            model = cls(json.loads(str(data['vocab'])),
                        embedding_dim=data['embedding'].shape[1],
                        hidden_dim=data['b1'].shape[0])
            model.params = {key: data[key].copy() for key in model.params}
        return model


class Adam:
    def __init__(self, params, learning_rate=0.02):
        self.lr, self.step = learning_rate, 0
        self.m = {key: np.zeros_like(value) for key, value in params.items()}
        self.v = {key: np.zeros_like(value) for key, value in params.items()}

    def update(self, params, gradients):
        self.step += 1
        for key in params:
            gradient = gradients[key]
            self.m[key] = 0.9 * self.m[key] + 0.1 * gradient
            self.v[key] = 0.999 * self.v[key] + 0.001 * gradient ** 2
            corrected_m = self.m[key] / (1 - 0.9 ** self.step)
            corrected_v = self.v[key] / (1 - 0.999 ** self.step)
            params[key] -= self.lr * corrected_m / (np.sqrt(corrected_v) + 1e-8)
