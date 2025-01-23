import torch.nn as nn
import torch.nn.functional as F

from model.layer import *
from model.base import PytorchLightningBase

from einops import rearrange
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class TimeXer(PytorchLightningBase):
    def __init__(self, args):
        super().__init__()
        self.input_len = args.input_len
        self.output_len = args.output_len
        self.hidden_dim = args.hidden_dim
        self.embedding_dim = args.embedding_dim
        self.lr = args.learning_rate
        self.num_heads = args.num_heads
        self.num_endo_vars = args.num_endo_vars
        self.num_exo_vars = args.num_exo_vars
        self.num_layers = args.num_layers
        self.save_hyperparameters()

        self.exogenous_encoder = ExogenousEncoder(self.hidden_dim, self.input_len)
        # self.endogenous_encoder = EndogenousEncoder(self.hidden_dim, self.input_len+1, self.num_endo_vars)
        self.endogenous_encoder = InvertedEndogenousEncoder(self.hidden_dim, self.input_len)
        self.temporal_feature_encoder = TemporalFeatureEncoder(self.embedding_dim)
        self.feature_fusion_network = FeatureFusionNetwork(self.embedding_dim, self.hidden_dim)

        decoder_layer = TransformerDecoderLayer(d_model=self.hidden_dim, nhead=self.num_heads, dim_feedforward=self.hidden_dim * 4, dropout=0.1)
        self.decoder = nn.TransformerDecoder(decoder_layer, self.num_layers)
        self.decoder_fc = nn.Sequential(
            nn.Linear(self.hidden_dim, self.output_len),
            nn.Dropout(0.2)
        )
    

    def split_inputs(self, inputs, meta_data):
        batch_size, num_vars = meta_data.shape
        endo_idx = torch.stack([meta_data[:,:27].argmax(dim=1),\
                                meta_data[:,27:34].argmax(dim=1)+27, \
                                meta_data[:,34:].argmax(dim=1)+34],axis=-1)
        gather_idx = endo_idx.unsqueeze(-1).expand(-1,-1, self.input_len)
        endo_inputs = inputs.gather(dim=1, index=gather_idx)

        mask = torch.ones((batch_size, num_vars), dtype=torch.bool)
        rows = torch.arange(batch_size).unsqueeze(-1).expand(-1, 3)
        mask[rows, endo_idx] = False
        exo_inputs =inputs[mask].view(batch_size, -1, 52)
        return endo_inputs, exo_inputs
    

    def forward(self, inputs, release_dates, image_embedding, text_embedding, meta_data):
        temporal_embedding = self.temporal_feature_encoder(release_dates)
        fusion_embedding = self.feature_fusion_network(image_embedding, text_embedding, temporal_embedding, meta_data)

        endo_inputs, exo_inputs = self.split_inputs(inputs, meta_data)
        endogenous_embedding = self.endogenous_encoder(endo_inputs, fusion_embedding.unsqueeze(1))
        exogenous_embedding = self.exogenous_encoder(exo_inputs)

        tgt = endogenous_embedding.unsqueeze(0)
        memory = exogenous_embedding
        decoder_out, attn_weights = self.decoder(tgt, memory)
        forecast = self.decoder_fc(decoder_out)

        return forecast.view(-1, self.output_len), attn_weights


    def phase_step(self, batch, phase):
        item_sales, inputs, release_dates, image_embeddings, text_embeddings, meta_data = batch
        forecasted_sales, _ = self.forward(inputs, release_dates, image_embeddings, text_embeddings, meta_data)
        if phase != 'predict':
            score = self.get_score(item_sales, forecasted_sales)
            score['loss'] = F.mse_loss(item_sales, forecasted_sales.squeeze())
            self.log_dict({f"{phase}_{k}":v for k,v in score.items()}, on_step=False, on_epoch=True)

            rescaled_score = self.get_score(item_sales * 1820, forecasted_sales * 1820)
            self.log_dict({f"{phase}_rescaled_{k}":v for k,v in rescaled_score.items()}, on_step=False, on_epoch=True)
            return score['loss']
        else:
            return forecasted_sales * 1820


class InvertedEndogenousEncoder(nn.Module):
    def __init__(self, embedding_dim, input_len):
        super().__init__()
        self.input_linear = TimeDistributed(nn.Linear(input_len, embedding_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=4, dropout=0.2)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, inputs, fusion_emb):
        inputs = inputs.permute(0,2,1)
        if inputs.dim() <= 2: inputs = inputs.unsqueeze(dim=-1)
        emb = self.input_linear(inputs)
        emb = torch.cat([emb, fusion_emb], dim=1)
        emb = self.encoder(emb)
        return emb[:,-1,:]


class EndogenousEncoder(nn.Module):
    def __init__(self, embedding_dim, input_len, num_vars):
        super().__init__()
        self.input_linear = TimeDistributed(nn.Linear(num_vars, embedding_dim))
        self.pos_embedding = PositionalEncoding(embedding_dim, max_len=input_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=4, dropout=0.2)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, inputs, fusion_emb):
        if inputs.dim() <= 2: inputs = inputs.unsqueeze(dim=-1)
        emb = self.input_linear(inputs)
        emb = torch.cat([emb, fusion_emb], dim=1)
        emb = self.pos_embedding(emb)
        emb = self.encoder(emb)
        return emb[:,-1,:]


class ExogenousEncoder(nn.Module):
    def __init__(self, embedding_dim, input_len):
        super().__init__()
        self.input_linear = TimeDistributed(nn.Linear(input_len, embedding_dim))

    def forward(self, inputs):
        inputs = inputs.permute(0,2,1)
        if inputs.dim() <= 2: inputs = inputs.unsqueeze(dim=-1)
        emb = self.input_linear(inputs)
        return emb