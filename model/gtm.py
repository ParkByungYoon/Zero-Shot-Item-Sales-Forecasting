import torch.nn as nn
import torch.nn.functional as F

from model.layer import *
from model.base import PytorchLightningBase
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class GTMTransformer(PytorchLightningBase):
    def __init__(self, input_len, output_len, num_vars, embedding_dim, hidden_dim, num_heads, num_layers, lr):
        super().__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.lr = lr
        self.save_hyperparameters()

        self.transformer_encoder = TransformerEncoder(hidden_dim, input_len, num_vars)
        self.temporal_feature_encoder = TemporalFeatureEncoder(embedding_dim)
        self.feature_fusion_network = FeatureFusionNetwork(embedding_dim, hidden_dim)

        decoder_layer = TransformerDecoderLayer(d_model=self.hidden_dim, nhead=num_heads, dim_feedforward=self.hidden_dim * 4, dropout=0.1)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.decoder_fc = nn.Sequential(
            nn.Linear(hidden_dim, self.output_len),
            nn.Dropout(0.2)
        )

    def forward(self, ntrends, release_dates, image_embedding, text_embedding, meta_data):
        encoder_embedding = self.transformer_encoder(ntrends)
        temporal_embedding = self.temporal_feature_encoder(release_dates)
        fusion_embedding = self.feature_fusion_network(image_embedding, text_embedding, temporal_embedding, meta_data)
            
        tgt = fusion_embedding.unsqueeze(0)
        memory = encoder_embedding
        decoder_out, attn_weights = self.decoder(tgt, memory)
        forecast = self.decoder_fc(decoder_out)

        return forecast.view(-1, self.output_len), attn_weights

    def phase_step(self, batch, phase):
        item_sales, ntrends, release_dates, image_embeddings, text_embeddings, meta_data = batch
        forecasted_sales, _ = self.forward(ntrends, release_dates, image_embeddings, text_embeddings, meta_data)
        forecasted_sales = forecasted_sales.sigmoid()
        if phase != 'predict':
            loss = F.mse_loss(item_sales, forecasted_sales.squeeze())
            adjusted_smape, r2_score = self.get_score(item_sales, forecasted_sales)
            self.log(f'{phase}_loss', loss)
            self.log(f'{phase}_adjusted_smape', adjusted_smape)
            self.log(f'{phase}_r2_score', r2_score)

            rescaled_adjusted_smape, rescaled_r2_score = self.get_score(item_sales * 1820, forecasted_sales * 1820)
            self.log(f'{phase}_rescaled_adjusted_smape', rescaled_adjusted_smape)
            self.log(f'{phase}_rescaled_r2_score', rescaled_r2_score)

            return loss
        else:
            return forecasted_sales * 1820


class GTMiTransformer(GTMTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transformer_encoder = InversedTransformerEncoder(self.hidden_dim, self.input_len)