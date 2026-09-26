
SPECIALS = ["<pad>", "<sos>", "<eos>", "<unk>"]
PAD, SOS, EOS, UNK = range(4)

def build_vocab(texts):
    return SPECIALS + sorted(set("".join(texts)))

def encode(text, stoi, add_sos_eos=False):
    ids = [stoi.get(ch, UNK) for ch in text]
    return [SOS] + ids + [EOS] if add_sos_eos else ids


def decode(ids, itos):
    return "".join(itos[i] for i in ids if i > UNK)



