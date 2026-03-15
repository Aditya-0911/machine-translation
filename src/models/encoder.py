import torch
import torch.nn as nn
from .embeddings import Embeddings
from config import hidden_dim,num_layers,embedding_dim

class Encoder(nn.Module):
    
    def __init__(self,vocab_size):
        super().__init__()

        self.embedding = Embeddings(vocab_size=vocab_size)

        self.lstm = nn.LSTM(
            batch_first=True,
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True
        )

    def forward(self, src):
    
        embedded = self.embedding(src)
        outputs, (hidden, cell) = self.lstm(embedded)

        # hidden shape: [num_layers*2, batch, hidden_dim]
        # Merge forward and backward by summing pairs
        # Result shape: [num_layers, batch, hidden_dim]
        hidden = hidden.view(num_layers, 2, -1, hidden_dim)
        hidden = (hidden[:, 0, :, :] + hidden[:, 1, :, :])  # sum forward + backward

        cell = cell.view(num_layers, 2, -1, hidden_dim)
        cell = (cell[:, 0, :, :] + cell[:, 1, :, :])

        return outputs, hidden, cell



