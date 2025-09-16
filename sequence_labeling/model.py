import sys
import torch
import torch.nn as nn
from transformers import AutoModel

class Net(nn.Module):
    def __init__(self, top_rnns=False, sense_size=None, dropout_rate=None, device='cpu', finetuning=False):
        super().__init__()
        self.roberta = AutoModel.from_pretrained('roberta-large')
        self.encoder_layers = nn.TransformerEncoderLayer(d_model=1024,dim_feedforward=2048, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layers, num_layers=6)
        self.sense_fc = nn.Linear(1024, sense_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.device = device
        self.finetuning = finetuning

    def forward(self, input_ids):
        '''
        x: (N, T). int64 sentence
        y: (N, T). int64 sense
        z: (N, T). int64 pos
        u: (N, T). int64 super_sense
        Returns
        enc: (N, T, VOCAB)
        '''
        input_ids = input_ids.to(self.device)
        output = self.roberta(input_ids)

        sequence_output = output[0]
#        sequence_output = self.dropout(sequence_output)
#        enc = self.transformer_encoder(enc)
        logits = self.sense_fc(sequence_output)

        return logits
