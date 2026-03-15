import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.tokenizer import Tokenizer
from src.data.dataset import TranslationDataset, collate_fn

from src.models.encoder import Encoder
from src.models.decoder import Decoder
from src.models.seq2seq import Seq2Seq

from config import batch_size, learning_rate, num_epochs

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    src_tokenizer = Tokenizer()
    tgt_tokenizer = Tokenizer()


    train_dataset = TranslationDataset(
        "train",
        src_tokenizer,
        tgt_tokenizer
    )
    
    print(f"Src vocab size: {len(src_tokenizer.word2idx)}")
    print(f"Tgt vocab size: {len(tgt_tokenizer.word2idx)}")

    val_dataset = TranslationDataset(
        "validation",
        src_tokenizer,
        tgt_tokenizer
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn, 
        num_workers=2
    )

    src_vocab_size = len(src_tokenizer.word2idx)
    tgt_vocab_size = len(tgt_tokenizer.word2idx)

    encoder = Encoder(src_vocab_size)
    decoder = Decoder(tgt_vocab_size)

    model = Seq2Seq(encoder, decoder, device).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=0)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    for epoch in range(num_epochs):

    # --- Training ---
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

        for src, dec_in, dec_out in progress_bar:
            src, dec_in, dec_out = src.to(device), dec_in.to(device), dec_out.to(device)
            optimizer.zero_grad()
            output = model(src, dec_in)
            output_dim = output.shape[-1]
            output = output[:,1:].reshape(-1, output_dim)
            dec_out = dec_out[:,1:].reshape(-1)
            loss = criterion(output, dec_out)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        # --- Validation ---
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for src, dec_in, dec_out in val_loader:
                src, dec_in, dec_out = src.to(device), dec_in.to(device), dec_out.to(device)
                output = model(src, dec_in)
                output_dim = output.shape[-1]
                output = output[:,1:].reshape(-1, output_dim)
                dec_out = dec_out[:,1:].reshape(-1)
                loss = criterion(output, dec_out)
                val_loss += loss.item()

        print(f"Epoch {epoch+1} | Train Loss: {total_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}")
    torch.save(model.state_dict(), "seq2seq_model.pt")

    import json

    with open("src_vocab.json", "w") as f:
        json.dump(src_tokenizer.word2idx, f)

    with open("tgt_vocab.json", "w") as f:
        json.dump(tgt_tokenizer.word2idx, f)

if __name__ == "__main__":
    main()