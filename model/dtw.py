import torch
import torch.nn as nn
import torch.nn.functional as F

from model.layer import *
from model.base import PytorchLightningBase
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class DTWPredictor(PytorchLightningBase):
    def __init__(self, embedding_dim, hidden_dim, lr):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.lr = lr
        self.save_hyperparameters()

        self.temporal_feature_encoder = TemporalFeatureEncoder(embedding_dim)
        self.feature_fusion_network = FeatureFusionNetwork(embedding_dim, hidden_dim)
        self.final_layer = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, center_items, neighbor_items):
        c_image, c_text, c_meta, c_date = center_items
        n_image, n_text, n_meta, n_date = neighbor_items

        c_temp = self.temporal_feature_encoder(c_date)
        c_item_embedding = self.feature_fusion_network(c_image, c_text, c_temp, c_meta)

        n_temp = self.temporal_feature_encoder(n_date)
        n_item_embedding = self.feature_fusion_network(n_image, n_text, n_temp, n_meta)
        
        return self.final_layer(torch.cat([c_item_embedding, n_item_embedding], dim=-1))

    def phase_step(self, batch, phase):
        dtw, center_items, neighbor_items = batch
        prediction = self.forward(center_items, neighbor_items)
        loss = F.mse_loss(dtw.squeeze(), prediction.squeeze())
        self.log(f'{phase}_loss', loss)
        return loss

class DTWDotProduct(DTWPredictor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, center_items, neighbor_items):
        c_image, c_text, c_meta, c_date = center_items
        n_image, n_text, n_meta, n_date = neighbor_items

        c_temp = self.temporal_feature_encoder(c_date)
        c_item_embedding = self.feature_fusion_network(c_image, c_text, c_temp, c_meta).unsqueeze(1)

        n_temp = self.temporal_feature_encoder(n_date)
        n_item_embedding = self.feature_fusion_network(n_image, n_text, n_temp, n_meta).unsqueeze(-1)
        return torch.bmm(c_item_embedding, n_item_embedding)