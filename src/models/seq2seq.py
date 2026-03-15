import torch
import torch.nn as nn
import random


class Seq2Seq(nn.Module):

    def __init__(self, encoder, decoder, device):

        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):

        batch_size = src.shape[0]
        tgt_len = tgt.shape[1]
        tgt_vocab_size = self.decoder.fc_out.out_features

        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)

        encoder_outputs, hidden, cell = self.encoder(src)

        input_token = tgt[:,0]

        for t in range(1, tgt_len):

            prediction, hidden, cell = self.decoder(
                input_token,
                hidden,
                cell,
                encoder_outputs
            )

            outputs[:,t] = prediction

            teacher_force = random.random() < teacher_forcing_ratio

            top1 = prediction.argmax(1)

            input_token = tgt[:,t] if teacher_force else top1

        return outputs