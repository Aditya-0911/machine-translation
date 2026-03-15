import torch
import torch.nn as nn

from .embeddings import Embeddings
from .attention import Attention
from config import embedding_dim, hidden_dim, num_layers

class Decoder(nn.Module):

    def __init__(self,vocab_size):
        super().__init__()
        self.embedding = Embeddings(vocab_size)
        self.attention = Attention()
        self.lstm = nn.LSTM(
            embedding_dim + hidden_dim*2,
            hidden_dim,
            num_layers,
            batch_first=True
        )
        self.fc_out = nn.Linear(hidden_dim,vocab_size)

    def forward(self,input_token,hidden,cell,encoder_outputs):

        input_token = input_token.unsqueeze(1)
        embedded = self.embedding(input_token)
        decoder_hidden = hidden[-1]
        attn_weights = self.attention(decoder_hidden, encoder_outputs)

        attn_weights = attn_weights.unsqueeze(1)
        context = torch.bmm(attn_weights, encoder_outputs)
        lstm_input = torch.cat((embedded, context), dim=2)
        output, (hidden, cell) = self.lstm(lstm_input, (hidden.contiguous(), cell.contiguous()))
        output = output.squeeze(1)

        output = output.squeeze(1)

        prediction = self.fc_out(output)

        return prediction, hidden, cell