import torch
import torch.nn as nn
from config import hidden_dim

class Attention(nn.Module):

    def __init__(self):

        super().__init__()

        self.W_enc = nn.Linear(hidden_dim*2,hidden_dim)
        self.W_dec = nn.Linear(hidden_dim, hidden_dim)

        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs):

        src_len = encoder_outputs.shape[1]

        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)

        energy = torch.tanh(
            self.W_enc(encoder_outputs) +
            self.W_dec(decoder_hidden)
        )

        scores = self.v(energy).squeeze(-1)

        attention_weights = torch.softmax(scores, dim=1)

        return attention_weights