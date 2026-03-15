import torch
import json

from src.data.tokenizer import Tokenizer
from src.models.encoder import Encoder
from src.models.decoder import Decoder
from src.models.seq2seq import Seq2Seq

from config import sos_idx, eos_idx


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def translate(sentence, model, src_tokenizer, tgt_tokenizer, max_len=20):

    model.eval()

    tokens = src_tokenizer.encode(sentence)

    src_tensor = torch.tensor(tokens).unsqueeze(0).to(device)

    with torch.no_grad():

        encoder_outputs, hidden, cell = model.encoder(src_tensor)

        input_token = torch.tensor([sos_idx]).to(device)

        outputs = []

        for _ in range(max_len):

            prediction, hidden, cell = model.decoder(
                input_token,
                hidden,
                cell,
                encoder_outputs
            )

            top1 = prediction.argmax(1)

            if top1.item() == eos_idx:
                break

            outputs.append(top1.item())

            input_token = top1

    return tgt_tokenizer.decode(outputs)


if __name__ == "__main__":

    src_tokenizer = Tokenizer()
    tgt_tokenizer = Tokenizer()

    with open("src_vocab.json") as f:
        src_tokenizer.word2idx = json.load(f)

    with open("tgt_vocab.json") as f:
        tgt_tokenizer.word2idx = json.load(f)

    src_tokenizer.idx2word = {v:k for k,v in src_tokenizer.word2idx.items()}
    tgt_tokenizer.idx2word = {v:k for k,v in tgt_tokenizer.word2idx.items()}

    # rebuild vocab (same as training)
    # from src.data.dataset import TranslationDataset

    # train_dataset = TranslationDataset("train", src_tokenizer, tgt_tokenizer)

    src_vocab_size = len(src_tokenizer.word2idx)
    tgt_vocab_size = len(tgt_tokenizer.word2idx)

    encoder = Encoder(src_vocab_size)
    decoder = Decoder(tgt_vocab_size)

    model = Seq2Seq(encoder, decoder, device).to(device)

    model.load_state_dict(torch.load("seq2seq_model.pt", map_location=device))

    model.eval()

    print("\nModel loaded. Type a sentence to translate.")
    print("Type 'quit' to exit.\n")

    while True:

        sentence = input("English > ")

        if sentence.lower() == "quit":
            break

        translation = translate(sentence, model, src_tokenizer, tgt_tokenizer)

        print("Translation >", translation)
        print()