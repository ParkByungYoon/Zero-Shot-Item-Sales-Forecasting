import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=104):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (
                    -math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TimeDistributed(nn.Module):
    # Takes any module and stacks the time dimension with the batch dimenison of inputs before applying the module
    # Insipired from https://keras.io/api/layers/recurrent_layers/time_distributed/
    # https://discuss.pytorch.org/t/any-pytorch-function-can-work-as-keras-timedistributed/1346/4
    def __init__(self, module, batch_first=True):
        super(TimeDistributed, self).__init__()
        self.module = module  # Can be any layer we wish to apply like Linear, Conv etc
        self.batch_first = batch_first

    def forward(self, x):
        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.permute(0,2,1).contiguous().view(-1, x.size(1))
        y = self.module(x_reshape)

        # We have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(
                -1))  # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1),
                       y.size(-1))  # (timesteps, samples, output_size)

        return y
    

class StaticFeatureEncoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, dropout=0.2):
        super(StaticFeatureEncoder, self).__init__()
        self.meta_linear = nn.Linear(50, embedding_dim)
        self.batchnorm = nn.BatchNorm1d(embedding_dim*3)
        self.feature_fusion = nn.Sequential(
            nn.Linear(embedding_dim*3, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, image_embedding, text_embedding, meta_data):
        meta_embedding = self.meta_linear(meta_data)
        features = torch.cat([image_embedding, text_embedding, meta_embedding], dim=1)
        features = self.batchnorm(features)
        features = self.feature_fusion(features)
        return features


class ItemSalesEncoder(nn.Module):
    def __init__(self, embedding_dim, input_len, num_items):
        super().__init__()
        self.input_linear = TimeDistributed(nn.Linear(num_items, embedding_dim))
        self.pos_embedding = PositionalEncoding(embedding_dim, max_len=input_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=4, dropout=0.2)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, input):
        if input.dim() <= 2: input = input.unsqueeze(dim=-1)
        emb = self.input_linear(input)
        emb = self.pos_embedding(emb)
        emb = self.encoder(emb)
        return emb

class InversedItemSalesEncoder(nn.Module):
    def __init__(self, embedding_dim, input_len):
        super().__init__()
        self.input_linear = TimeDistributed(nn.Linear(input_len, embedding_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=4, dropout=0.2)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, input):

        input = input.permute(0,2,1)

        if input.dim() <= 2: input = input.unsqueeze(dim=-1)
        emb = self.input_linear(input)
        emb = self.encoder(emb)
        return emb


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerDecoderLayer, self).__init__()
        
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = F.relu

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerDecoderLayer, self).__setstate__(state)

    def forward(self, tgt, memory, tgt_mask = None, memory_mask = None, tgt_key_padding_mask = None, 
            memory_key_padding_mask = None, tgt_is_causal = None, memory_is_causal=False):
        if type(tgt) is tuple: tgt = tgt[0]
        tgt2, attn_weights = self.multihead_attn(tgt, memory.permute(1,0,2), memory.permute(1,0,2))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, attn_weights