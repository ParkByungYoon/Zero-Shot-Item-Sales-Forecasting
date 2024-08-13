from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.regression import R2Score, SymmetricMeanAbsolutePercentageError

from model.layer import *
from model.base import PytorchLightningBase
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class KNNTransformerWaveform(PytorchLightningBase):
    def __init__(self, input_len, output_len, num_neighbors, embedding_dim, hidden_dim, num_heads, num_layers, lr):
        super().__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.lr = lr
        self.save_hyperparameters()

        self.item_sales_encoder = ItemSalesEncoder(hidden_dim, input_len, num_neighbors)
        self.temporal_feature_encoder = TemporalFeatureEncoder(embedding_dim)
        self.feature_fusion_network = FeatureFusionNetwork(embedding_dim, hidden_dim)

        # Decoder
        decoder_layer = TransformerDecoderLayer(d_model=self.hidden_dim, nhead=num_heads, \
                                                dim_feedforward=self.hidden_dim * 4, dropout=0.1)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.decoder_fc = nn.Sequential(
            nn.Linear(hidden_dim, self.output_len),
            nn.Dropout(0.2)
        )

    def forward(self, k_item_sales, release_dates, image_embedding, text_embedding, meta_data):
        # Encode features and get inputs
        k_item_sales_embedding = self.item_sales_encoder(k_item_sales)

        # Fuse static features together
        temporal_embedding = self.temporal_feature_encoder(release_dates)
        fusion_embedding = self.feature_fusion_network(image_embedding, text_embedding, temporal_embedding, meta_data)
            
        tgt = fusion_embedding.unsqueeze(0)
        memory = k_item_sales_embedding
        decoder_out, attn_weights = self.decoder(tgt, memory)
        forecast = self.decoder_fc(decoder_out)

        return forecast.view(-1, self.output_len), attn_weights

    def phase_step(self, batch, phase):
        item_sales, mu_sigma, k_item_sales, _, release_dates, image_embeddings, text_embeddings, meta_data = batch
        forecasted_sales, _ = self.forward(k_item_sales, release_dates, image_embeddings, text_embeddings, meta_data)
        
        loss = F.mse_loss(item_sales, forecasted_sales.squeeze())
        adjusted_smape, r2_score = self.get_score(item_sales, forecasted_sales)
        self.log(f'{phase}_loss', loss)
        self.log(f'{phase}_adjusted_smape', adjusted_smape)
        self.log(f'{phase}_r2_score', r2_score)

        rescaled_item_sales = item_sales * mu_sigma[:,1].unsqueeze(dim=-1) + mu_sigma[:,0].unsqueeze(dim=-1)
        rescaled_forecasted_sales = forecasted_sales * mu_sigma[:,1].unsqueeze(dim=-1) + mu_sigma[:,0].unsqueeze(dim=-1)
        rescaled_adjusted_smape, rescaled_r2_score = self.get_score(rescaled_item_sales, rescaled_forecasted_sales)

        self.log(f'{phase}_rescaled_adjusted_smape', rescaled_adjusted_smape)
        self.log(f'{phase}_rescaled_r2_score', rescaled_r2_score)

        return loss


class KNNiTransformerWaveform(KNNTransformerWaveform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.item_sales_encoder = InversedItemSalesEncoder(self.hidden_dim, self.input_len)