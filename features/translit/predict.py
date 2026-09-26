"""Arabic -> Arabizi inference in plain NumPy, so the server doesn't need PyTorch.

Weights come from models/translit.npz (made by `python -m features.translit.export`).
The math mirrors model.py exactly: 3-layer LSTM encoder, LSTM decoder with dot-product attention, greedy decoding.
"""
import numpy as np

from features.translit.vocab import encode, decode, SOS, EOS

W = dict(np.load("models/translit.npz"))
src_vocab, tgt_vocab = [str(t) for t in W.pop("src_vocab")], [str(t) for t in W.pop("tgt_vocab")]
src_stoi = {ch: i for i, ch in enumerate(src_vocab)}
LAYERS = sum(k.startswith("enc.lstm.weight_ih_l") for k in W)
HIDDEN = W["enc.lstm.weight_hh_l0"].shape[1]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def lstm_step(prefix, x, h, c):
    """One time step through every layer. h, c: (LAYERS, HIDDEN). Returns the top layer's output."""
    h, c = h.copy(), c.copy()
    for l in range(LAYERS):
        gates = (W[f"{prefix}.weight_ih_l{l}"] @ x + W[f"{prefix}.bias_ih_l{l}"]
                 + W[f"{prefix}.weight_hh_l{l}"] @ h[l] + W[f"{prefix}.bias_hh_l{l}"])
        i, f, g, o = np.split(gates, 4)  # PyTorch gate order: input, forget, cell, output
        c[l] = sigmoid(f) * c[l] + sigmoid(i) * np.tanh(g)
        h[l] = sigmoid(o) * np.tanh(c[l])
        x = h[l]  # this layer's output feeds the next layer
    return x, h, c


def translate(text, max_len=200):
    text = text.strip()
    if not text:
        return ""

    # encoder: read the Arabic one character at a time, keep every top-layer output for attention
    h = np.zeros((LAYERS, HIDDEN), dtype=np.float32)
    c = np.zeros((LAYERS, HIDDEN), dtype=np.float32)
    enc_outputs = []
    for idx in encode(text, src_stoi):
        top, h, c = lstm_step("enc.lstm", W["enc.embeddings.weight"][idx], h, c)
        enc_outputs.append(top)
    enc_outputs = np.stack(enc_outputs)  # (T, HIDDEN)

    # decoder: start from the encoder's final state, feed back its own prediction each step
    token, out = SOS, []
    for _ in range(max_len):
        top, h, c = lstm_step("dec.lstm", W["dec.embedding.weight"][token], h, c)
        scores = enc_outputs @ top                      # how well each Arabic char matches the current state
        weights = np.exp(scores - scores.max())
        weights /= weights.sum()                        # softmax
        context = weights @ enc_outputs                 # where the decoder is "looking"
        logits = W["dec.linear.weight"] @ np.concatenate([top, context]) + W["dec.linear.bias"]
        token = int(logits.argmax())
        if token == EOS:
            break
        out.append(token)
    return decode(out, tgt_vocab)
