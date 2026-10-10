"""Arabic <-> Arabizi inference in plain NumPy, so the server doesn't need PyTorch.

Weights come from models/translit_<direction>.npz (made by `python -m features.translit.export <direction>`).
The math mirrors model.py exactly: bidirectional LSTM encoder, bridge layers into the decoder,
LSTM decoder with dot-product attention, greedy decoding. Both directions use the same code,
only the weights and the input cleaning differ.
"""
import re
from pathlib import Path

import numpy as np

from features.normalize import normalize_arabic, normalize_arabizi
from features.translit.vocab import encode, decode, SOS, EOS

# input cleaning per direction: same functions the training data went through
NORMALIZE = {"ar2az": normalize_arabic, "az2ar": normalize_arabizi}


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class Translator:
    def __init__(self, path):
        self.W = dict(np.load(path))
        self.src_vocab = [str(t) for t in self.W.pop("src_vocab")]
        self.tgt_vocab = [str(t) for t in self.W.pop("tgt_vocab")]
        self.src_stoi = {ch: i for i, ch in enumerate(self.src_vocab)}
        self.layers = sum(bool(re.fullmatch(r"enc\.lstm\.weight_ih_l\d+", k)) for k in self.W)  # skips *_reverse
        self.hidden = self.W["enc.lstm.weight_hh_l0"].shape[1]

    def linear(self, name, x):
        return self.W[f"{name}.weight"] @ x + self.W[f"{name}.bias"]

    def lstm_cell(self, prefix, suffix, x, h, c):
        """One LSTM step for one layer / direction. suffix is 'l0', 'l0_reverse', 'l1', ..."""
        W = self.W
        gates = (W[f"{prefix}.weight_ih_{suffix}"] @ x + W[f"{prefix}.bias_ih_{suffix}"]
                 + W[f"{prefix}.weight_hh_{suffix}"] @ h + W[f"{prefix}.bias_hh_{suffix}"])
        i, f, g, o = np.split(gates, 4)  # PyTorch gate order: input, forget, cell, output
        c = sigmoid(f) * c + sigmoid(i) * np.tanh(g)
        h = sigmoid(o) * np.tanh(c)
        return h, c

    def run_direction(self, suffix, xs):
        """One encoder layer in one direction over the sequence. Returns per-step outputs and final (h, c)."""
        h = c = np.zeros(self.hidden, dtype=np.float32)
        outs = []
        for x in xs:
            h, c = self.lstm_cell("enc.lstm", suffix, x, h, c)
            outs.append(h)
        return outs, h, c

    def encode_sentence(self, ids):
        """Bidirectional encoder -> (encoder outputs projected to hidden, decoder start h, c)."""
        xs = [self.W["enc.embeddings.weight"][i] for i in ids]
        h0, c0 = [], []
        for l in range(self.layers):
            fwd, hf, cf = self.run_direction(f"l{l}", xs)
            bwd, hb, cb = self.run_direction(f"l{l}_reverse", xs[::-1])  # reads right to left
            bwd = bwd[::-1]                                              # back into left-to-right order
            xs = [np.concatenate([f, b]) for f, b in zip(fwd, bwd)]      # next layer sees both directions
            # bridge: squeeze [forward; backward] final states into one decoder state per layer
            h0.append(np.tanh(self.linear("enc.hidden_bridge", np.concatenate([hf, hb]))))
            c0.append(np.tanh(self.linear("enc.cell_bridge", np.concatenate([cf, cb]))))
        enc_outputs = np.stack(xs) @ self.W["dec.enc_projection.weight"].T + self.W["dec.enc_projection.bias"]
        return enc_outputs, h0, c0

    def translate(self, text, max_len=200):
        """text must already be normalized for this direction."""
        if not text:
            return ""
        enc_outputs, h, c = self.encode_sentence(encode(text, self.src_stoi))

        # decoder: start from the bridged encoder state, feed back its own prediction each step
        token, out = SOS, []
        for _ in range(max_len):
            x = self.W["dec.embedding.weight"][token]
            for l in range(self.layers):
                h[l], c[l] = self.lstm_cell("dec.lstm", f"l{l}", x, h[l], c[l])
                x = h[l]
            top = x
            scores = enc_outputs @ top                      # how well each input char matches the current state
            weights = np.exp(scores - scores.max())
            weights /= weights.sum()                        # softmax
            context = weights @ enc_outputs                 # where the decoder is "looking"
            logits = self.linear("dec.linear", np.concatenate([top, context]))
            token = int(logits.argmax())
            if token == EOS:
                break
            out.append(token)
        return decode(out, self.tgt_vocab)


# every exported direction in models/ is served; a missing one is just not offered
TRANSLATORS = {d: Translator(f"models/translit_{d}.npz") for d in NORMALIZE if Path(f"models/translit_{d}.npz").exists()}


def translate(text, direction="ar2az"):
    return TRANSLATORS[direction].translate(NORMALIZE[direction](text))
