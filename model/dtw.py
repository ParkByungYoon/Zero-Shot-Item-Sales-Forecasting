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
        temporal_embedding = self.temporal_feature_encoder(center_items[3])
        center_item_embedding = self.feature_fusion_network(center_items[0], center_items[1], temporal_embedding, center_items[2])
        neighbor_item_embedding = self.feature_fusion_network(neighbor_items[0], neighbor_items[1], temporal_embedding, neighbor_items[2])
        concatenated = torch.cat([center_item_embedding, neighbor_item_embedding], dim=-1)
        return self.final_layer(concatenated)

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
        temporal_embedding = self.temporal_feature_encoder(center_items[3])
        center_item_embedding = self.feature_fusion_network(center_items[0], center_items[1], temporal_embedding, center_items[2])
        neighbor_item_embedding = self.feature_fusion_network(neighbor_items[0], neighbor_items[1], temporal_embedding, neighbor_items[2])
        return torch.bmm(center_item_embedding.unsqueeze(1), neighbor_item_embedding.unsqueeze(-1))