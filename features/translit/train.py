import os
import random

import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

from features.translit.model import Encoder, Decoder, Seq2Seq
from features.translit.vocab import build_vocab, encode



data = pd.read_json("data/preprocessed_data.jsonl", lines=True)
data=data[['src','tgt']]

data["src"] = data["src"].str.strip()
data["tgt"] = data["tgt"].str.strip()
data = data[(data.src.str.len().between(1, 200)) & (data.tgt.str.len().between(1, 200))]



src_vocab = build_vocab(data.src)
tgt_vocab = build_vocab(data.tgt)


src_stoi = {token:i for i, token in enumerate(src_vocab)}
tgt_stoi = {token:i for i, token in enumerate(tgt_vocab)}


#building the dataloaders
class TranslitDataset(Dataset):
    def __init__(self, df):
        self.src = df["src"].tolist()
        self.tgt = df["tgt"].tolist()

    def __len__(self):
        return len(self.src)

    def __getitem__(self, i):
        src_ids = encode(self.src[i], src_stoi,  add_sos_eos=False)   # Arabic, no specials
        tgt_ids = encode(self.tgt[i], tgt_stoi, add_sos_eos=True)    # Arabizi, with <sos>/<eos>
        return torch.tensor(src_ids), torch.tensor(tgt_ids)

def collate(batch):
    src_list, tgt_list = zip(*batch)
    src = pad_sequence(src_list, batch_first=True, padding_value=0)
    tgt = pad_sequence(tgt_list, batch_first=True, padding_value=0)
    return src, tgt





random.seed(42); torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG = dict(emb=128, hidden=512, layers=1, dropout=0.1)
EPOCHS, LR, BATCH_SIZE =   60, 1e-3, 64

train_df = data.sample(frac=0.9, random_state=42)
val_df = data.drop(train_df.index)
train_loader = DataLoader(TranslitDataset(train_df), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)
val_loader = DataLoader(TranslitDataset(val_df), batch_size=BATCH_SIZE, collate_fn=collate)


c = CONFIG
model = Seq2Seq(Encoder(len(src_vocab), c["emb"], c["hidden"], c["layers"], c["dropout"]),
                Decoder(len(tgt_vocab), c["emb"], c["hidden"], c["layers"], c["dropout"])).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss(ignore_index=0)



def run_epoch(loader, train):
    model.train(train)
    total = 0
    with torch.set_grad_enabled(train):
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            out = model(src, tgt, 0.5 if train else 0.0)
            loss = criterion(out[:, 1:].reshape(-1, out.shape[-1]), tgt[:, 1:].reshape(-1))
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total += loss.item()
    return total / len(loader)

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    for epoch in range(1, EPOCHS + 1):
        tr, va = run_epoch(train_loader, True), run_epoch(val_loader, False)
        print(f"epoch {epoch:2d} | train {tr:.3f} | val {va:.3f}")
    torch.save({"model": model.state_dict(), "src_vocab": src_vocab,
                "tgt_vocab": tgt_vocab, "config": CONFIG}, "models/translit.pt")
    print("saved models/translit.pt")