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

    def forward(self,src):
        
        embedded = self.embedding(src)

        outputs,(hidden,cell) = self.lstm(embedded)

        return outputs,hidden,cell



