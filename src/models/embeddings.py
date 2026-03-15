import torch
import torch.nn as nn
from config import embedding_dim, pad_idx

class Embeddings(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, 
            embedding_dim=embedding_dim, 
            padding_idx=pad_idx
        )

    def forward(self,x):
        return self.embedding(x)