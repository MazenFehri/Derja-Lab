import torch
import torch.nn as nn
import random
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from features.translit.vocab import SOS, EOS


#builing the encoder 
class Encoder(nn.Module):
    def __init__(self,vocab_size,emb_size,hidden_size,num_layers,dropout):
        super().__init__()

        self.embeddings = nn.Embedding(vocab_size,emb_size,padding_idx=0)
        self.dropout = nn.Dropout(p=dropout)

        self.lstm = nn.LSTM(input_size=emb_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            bidirectional=True,
                            dropout=dropout if num_layers>1 else 0,
                            )              


        #making a bridge because the encoder has a bidirectional lstm
        # Combine forward + backward states
        self.hidden_bridge = nn.Linear(
            hidden_size * 2,
            hidden_size
        )

        self.cell_bridge = nn.Linear(
            hidden_size * 2,
            hidden_size
        )

    def forward(self, src):

        emb = self.embeddings(src)
        embedded = self.dropout(emb)

        # pack so the backward direction starts at each sentence's last real char, not at the <pad>s
        lengths = (src != 0).sum(1).cpu()
        packed = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        outputs, (h_n, c_n) = self.lstm(packed)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True, total_length=src.size(1))

        # h_n:
        # [num_layers * 2, batch, hidden_size]

        # Separate forward and backward states
        h_forward = h_n[0::2]
        h_backward = h_n[1::2]

        c_forward = c_n[0::2]
        c_backward = c_n[1::2]

        # [num_layers, batch, hidden_size * 2]
        h = torch.cat((h_forward, h_backward), dim=2)
        c = torch.cat((c_forward, c_backward), dim=2)

        # Project back to decoder hidden size
        h = torch.tanh(self.hidden_bridge(h))
        c = torch.tanh(self.cell_bridge(c))

        return outputs, (h, c)


#building the decoder 
class Decoder(nn.Module):
    def __init__(self,vocab_size,emb_size,hidden_size,num_layers,dropout):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size,emb_size,padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(input_size=emb_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers>1 else 0) 

        self.linear = nn.Linear(2*hidden_size,vocab_size)
        self.enc_projection = nn.Linear(
            hidden_size * 2,
            hidden_size)

    def forward(self,input,hidden,cell,enc_outputs,mask):
        # enc_outputs arrive already projected to hidden_size (done once per sentence in Seq2Seq)
        input = input.unsqueeze(1)
        embedded = self.dropout(self.embedding(input))
        output, (h,c) = self.lstm(embedded,(hidden,cell))
        h_top = output.squeeze(1)
        
        scores =torch.bmm(enc_outputs, h_top.unsqueeze(2)).squeeze(2) 
        weights = torch.softmax(scores.masked_fill(~mask, float("-inf")), dim=1)
        context = torch.bmm(weights.unsqueeze(1), enc_outputs).squeeze(1)
        logits = self.linear(torch.cat([h_top,context],dim=1))
        return logits , (h,c) , weights



#Building the seq2seq model
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.enc = encoder
        self.dec = decoder

    def forward(self, src, tgt,teacher_forcing_ratio=0.5):         
        B, T = tgt.shape
        V = self.dec.linear.out_features                        

        outputs = torch.zeros(B, T, V, device=src.device)          

        enc_outputs, (h, c) = self.enc(src)
        enc_outputs = self.dec.enc_projection(enc_outputs)   # once, not on every decoder step
        mask = src != 0
        input = tgt[:, 0]                                 
        for t in range(1, T):                            
            logits, (h, c),weights = self.dec(input, h, c,enc_outputs,mask)                
            outputs[:, t] = logits                              

            use_teacher = random.random() < teacher_forcing_ratio
            input = tgt[:, t] if use_teacher else logits.argmax(1) 

        return outputs


    @torch.no_grad()
    def translate(self, src, max_len=200):
        enc_outputs, (h, c) = self.enc(src)
        enc_outputs = self.dec.enc_projection(enc_outputs)
        mask = src != 0
        input, out = torch.tensor([SOS], device=src.device), []
        for _ in range(max_len):
            logits, (h, c), _ = self.dec(input, h, c, enc_outputs, mask)
            input = logits.argmax(1)
            if input.item() == EOS: break
            out.append(input.item())
        return out