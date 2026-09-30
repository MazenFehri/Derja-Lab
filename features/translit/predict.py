"""Arabic -> Arabizi inference in plain NumPy, so the server doesn't need PyTorch.

Weights come from models/translit.npz (made by `python -m features.translit.export`).
The math mirrors model.py exactly: bidirectional LSTM encoder, bridge layers into the decoder,
LSTM decoder with dot-product attention, greedy decoding.
"""
import re

import numpy as np

from features.translit.vocab import encode, decode, SOS, EOS

W = dict(np.load("models/translit.npz"))
src_vocab, tgt_vocab = [str(t) for t in W.pop("src_vocab")], [str(t) for t in W.pop("tgt_vocab")]
src_stoi = {ch: i for i, ch in enumerate(src_vocab)}
LAYERS = sum(bool(re.fullmatch(r"enc\.lstm\.weight_ih_l\d+", k)) for k in W)  # skips the *_reverse weights
HIDDEN = W["enc.lstm.weight_hh_l0"].shape[1]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def linear(name, x):
    return W[f"{name}.weight"] @ x + W[f"{name}.bias"]


def lstm_cell(prefix, suffix, x, h, c):
    """One LSTM step for one layer / direction. suffix is 'l0', 'l0_reverse', 'l1', ..."""
    gates = (W[f"{prefix}.weight_ih_{suffix}"] @ x + W[f"{prefix}.bias_ih_{suffix}"]
             + W[f"{prefix}.weight_hh_{suffix}"] @ h + W[f"{prefix}.bias_hh_{suffix}"])
    i, f, g, o = np.split(gates, 4)  # PyTorch gate order: input, forget, cell, output
    c = sigmoid(f) * c + sigmoid(i) * np.tanh(g)
    h = sigmoid(o) * np.tanh(c)
    return h, c


def run_direction(suffix, xs):
    """Run one encoder layer in one direction over the sequence. Returns per-step outputs and final (h, c)."""
    h = c = np.zeros(HIDDEN, dtype=np.float32)
    outs = []
    for x in xs:
        h, c = lstm_cell("enc.lstm", suffix, x, h, c)
        outs.append(h)
    return outs, h, c


def encode_sentence(ids):
    """Bidirectional encoder -> (encoder outputs projected to HIDDEN, decoder start h, c)."""
    xs = [W["enc.embeddings.weight"][i] for i in ids]
    h0, c0 = [], []
    for l in range(LAYERS):
        fwd, hf, cf = run_direction(f"l{l}", xs)
        bwd, hb, cb = run_direction(f"l{l}_reverse", xs[::-1])  # reads right to left
        bwd = bwd[::-1]                                          # back into left-to-right order
        xs = [np.concatenate([f, b]) for f, b in zip(fwd, bwd)]  # next layer sees both directions
        # bridge: squeeze [forward; backward] final states into one decoder state per layer
        h0.append(np.tanh(linear("enc.hidden_bridge", np.concatenate([hf, hb]))))
        c0.append(np.tanh(linear("enc.cell_bridge", np.concatenate([cf, cb]))))
    enc_outputs = np.stack(xs) @ W["dec.enc_projection.weight"].T + W["dec.enc_projection.bias"]  # (T, HIDDEN)
    return enc_outputs, np.stack(h0), np.stack(c0)


def translate(text, max_len=200):
    text = text.strip()
    if not text:
        return ""

    enc_outputs, h, c = encode_sentence(encode(text, src_stoi))

    # decoder: start from the bridged encoder state, feed back its own prediction each step
    token, out = SOS, []
    h, c = list(h), list(c)
    for _ in range(max_len):
        x = W["dec.embedding.weight"][token]
        for l in range(LAYERS):
            h[l], c[l] = lstm_cell("dec.lstm", f"l{l}", x, h[l], c[l])
            x = h[l]
        top = x
        scores = enc_outputs @ top                      # how well each Arabic char matches the current state
        weights = np.exp(scores - scores.max())
        weights /= weights.sum()                        # softmax
        context = weights @ enc_outputs                 # where the decoder is "looking"
        logits = linear("dec.linear", np.concatenate([top, context]))
        token = int(logits.argmax())
        if token == EOS:
            break
        out.append(token)
    return decode(out, tgt_vocab)
