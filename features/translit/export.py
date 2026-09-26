"""Convert models/translit.pt (PyTorch) into models/translit.npz (NumPy) for the server.

Run after training:  python -m features.translit.export
Needs PyTorch, so it runs on your machine, not on the server.
"""
import numpy as np
import pandas as pd
import torch

from features.translit.model import Encoder, Decoder, Seq2Seq
from features.translit.vocab import encode, decode

ckpt = torch.load("models/translit.pt", map_location="cpu")
np.savez_compressed(
    "models/translit.npz",
    **{k: v.numpy() for k, v in ckpt["model"].items()},
    src_vocab=np.array(ckpt["src_vocab"]),
    tgt_vocab=np.array(ckpt["tgt_vocab"]),
)
print("saved models/translit.npz")

# check: the NumPy model must translate exactly like the PyTorch one
from features.translit import predict  # imported here so it loads the file we just wrote

c = ckpt["config"]
model = Seq2Seq(Encoder(len(ckpt["src_vocab"]), c["emb"], c["hidden"], c["layers"], c["dropout"]),
                Decoder(len(ckpt["tgt_vocab"]), c["emb"], c["hidden"], c["layers"], c["dropout"]))
model.load_state_dict(ckpt["model"])
model.eval()
stoi = {ch: i for i, ch in enumerate(ckpt["src_vocab"])}

samples = pd.read_json("data/preprocessed_data.jsonl", lines=True)["src"].str.strip().head(100)
samples = [s for s in samples if 0 < len(s) <= 200]
mismatches = []
for s in samples:
    expected = decode(model.translate(torch.tensor([encode(s, stoi)])), ckpt["tgt_vocab"])
    if predict.translate(s) != expected:
        mismatches.append((s, expected, predict.translate(s)))

for s, want, got in mismatches[:5]:
    print(f"MISMATCH {s!r}\n  torch: {want!r}\n  numpy: {got!r}")
assert not mismatches, f"{len(mismatches)}/{len(samples)} sentences differ between PyTorch and NumPy"
print(f"check ok: NumPy matches PyTorch on {len(samples)} sentences")
