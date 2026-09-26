import torch
from features.translit.model import Encoder, Decoder, Seq2Seq
from features.translit.vocab import encode, decode

ckpt = torch.load("models/translit.pt", map_location="cpu")
src_vocab, tgt_vocab, c = ckpt["src_vocab"], ckpt["tgt_vocab"], ckpt["config"]
src_stoi = {ch: i for i, ch in enumerate(src_vocab)}

model = Seq2Seq(Encoder(len(src_vocab), c["emb"], c["hidden"], c["layers"], c["dropout"]),
                Decoder(len(tgt_vocab), c["emb"], c["hidden"], c["layers"], c["dropout"]))
model.load_state_dict(ckpt["model"])
model.eval()

def translate(text):
    text = text.strip()
    if not text:
        return ""
    return decode(model.translate(torch.tensor([encode(text, src_stoi)])), tgt_vocab)
