import torch
import torch.nn as nn
import random
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
                            dropout=dropout if num_layers>1 else 0)              



    def forward(self,src) : 
        emb = self.embeddings(src)
        embedded = self.dropout(emb)
        outputs , (h_n , c_n) = self.lstm(embedded)

        return outputs , (h_n , c_n) 


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

    def forward(self,input,hidden,cell,enc_outputs,mask):
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
        mask = src != 0
        input, out = torch.tensor([SOS], device=src.device), []
        for _ in range(max_len):
            logits, (h, c), _ = self.dec(input, h, c, enc_outputs, mask)
            input = logits.argmax(1)
            if input.item() == EOS: break
            out.append(input.item())
        return out