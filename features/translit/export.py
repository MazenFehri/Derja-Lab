"""Convert models/translit_<direction>.pt (PyTorch) into models/translit_<direction>.npz (NumPy) for the server.

Run after training:  python -m features.translit.export ar2az   (or az2ar)
Needs PyTorch, so it runs on your machine, not on the server.
"""
import sys

import numpy as np
import pandas as pd
import torch

from features.translit.model import Encoder, Decoder, Seq2Seq
from features.translit.predict import Translator
from features.translit.vocab import encode, decode

DIRECTION = sys.argv[1] if len(sys.argv) > 1 else "ar2az"
assert DIRECTION in ("ar2az", "az2ar"), "direction must be ar2az or az2ar"
PT, NPZ = f"models/translit_{DIRECTION}.pt", f"models/translit_{DIRECTION}.npz"

ckpt = torch.load(PT, map_location="cpu")
np.savez_compressed(
    NPZ,
    **{k: v.numpy() for k, v in ckpt["model"].items()},
    src_vocab=np.array(ckpt["src_vocab"]),
    tgt_vocab=np.array(ckpt["tgt_vocab"]),
)
print(f"saved {NPZ}")

# check: the NumPy model must translate exactly like the PyTorch one
c = ckpt["config"]
model = Seq2Seq(Encoder(len(ckpt["src_vocab"]), c["emb"], c["hidden"], c["layers"], c["dropout"]),
                Decoder(len(ckpt["tgt_vocab"]), c["emb"], c["hidden"], c["layers"], c["dropout"]))
model.load_state_dict(ckpt["model"])
model.eval()
stoi = {ch: i for i, ch in enumerate(ckpt["src_vocab"])}
numpy_model = Translator(NPZ)

column = "src" if DIRECTION == "ar2az" else "tgt"  # the input side for this direction
samples = pd.read_json("data/preprocessed_data.jsonl", lines=True)[column].str.strip().head(100)
samples = [s for s in samples if 0 < len(s) <= 200]
mismatches = []
for s in samples:
    expected = decode(model.translate(torch.tensor([encode(s, stoi)])), ckpt["tgt_vocab"])
    got = numpy_model.translate(s)
    if got != expected:
        mismatches.append((s, expected, got))

for s, want, got in mismatches[:5]:
    print(f"MISMATCH {s!r}\n  torch: {want!r}\n  numpy: {got!r}")
assert not mismatches, f"{len(mismatches)}/{len(samples)} sentences differ between PyTorch and NumPy"
print(f"check ok: NumPy matches PyTorch on {len(samples)} sentences")
