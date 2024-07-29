import torch
import torch.nn as nn
import torch.nn.functional as F

from model.layer import *
from model.base import PytorchLightningBase
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class KNNTransformerMuSigma(PytorchLightningBase):
    def __init__(self, input_len, output_len, num_neighbors, embedding_dim, hidden_dim, lr):
        super().__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.lr = lr
        self.save_hyperparameters()

        self.item_sales_encoder = ItemSalesEncoder(hidden_dim, input_len, num_neighbors)
        self.static_feature_encoder = StaticFeatureEncoder(embedding_dim, hidden_dim)
        self.final_layer = nn.Sequential(
            nn.Linear(hidden_dim*3, self.output_len),
            nn.Dropout(0.2)
        )


    def forward(self, k_item_mu_sigma, image_embedding, text_embedding, meta_data):
        k_item_sales_embedding = self.item_sales_encoder(k_item_mu_sigma)
        static_feature_fusion = self.static_feature_encoder(image_embedding, text_embedding, meta_data)
        concatenated = torch.cat([k_item_sales_embedding.flatten(1), static_feature_fusion], dim=1)
        return self.final_layer(concatenated)
    

    def phase_step(self, batch, phase):
        _, mu_sigma, _, k_item_mu_sigma, image_embeddings, text_embeddings, meta_data = batch
        forecasted_mu_sigma = self.forward(k_item_mu_sigma, image_embeddings, text_embeddings, meta_data)
        
        loss = F.mse_loss(mu_sigma, forecasted_mu_sigma)
        adjusted_smape, r2_score = self.get_score(mu_sigma, forecasted_mu_sigma)

        self.log(f'{phase}_loss', loss)
        self.log(f'{phase}_adjusted_smape', adjusted_smape)
        self.log(f'{phase}_r2_score', r2_score)

        return loss, adjusted_smape, r2_score
    

class KNNFeatureFusionMuSigma(PytorchLightningBase):
    def __init__(self, input_len, output_len, num_neighbors, embedding_dim, hidden_dim, lr):
        super().__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.lr = lr
        self.save_hyperparameters()

        self.mu_encoder = nn.Sequential(
            nn.Linear(num_neighbors, hidden_dim),
            nn.ReLU()
        )
        self.sigma_encoder = nn.Sequential(
            nn.Linear(num_neighbors, hidden_dim),
            nn.ReLU()
        )
        self.static_feature_encoder = StaticFeatureEncoder(embedding_dim, hidden_dim)
        self.final_layer = nn.Sequential(
            nn.Linear(hidden_dim*3, self.output_len),
            nn.Dropout(0.2)
        )


    def forward(self, k_item_mu_sigma, image_embedding, text_embedding, meta_data):
        mu_embedding = self.mu_encoder(k_item_mu_sigma[:,:,0])
        sigma_embedding = self.sigma_encoder(k_item_mu_sigma[:,:,1])
        static_feature_fusion = self.static_feature_encoder(image_embedding, text_embedding, meta_data)
        concatenated = torch.cat([mu_embedding, sigma_embedding, static_feature_fusion], dim=1)
        return self.final_layer(concatenated)
    

    def phase_step(self, batch, phase):
        _, mu_sigma, _, k_item_mu_sigma, image_embeddings, text_embeddings, meta_data = batch
        forecasted_mu_sigma = self.forward(k_item_mu_sigma, image_embeddings, text_embeddings, meta_data)
        
        mu_loss = F.mse_loss(mu_sigma[:,0], forecasted_mu_sigma[:,0])
        sigma_loss = F.mse_loss(mu_sigma[:,1], forecasted_mu_sigma[:,1])
        loss = mu_loss + sigma_loss
        self.log(f'{phase}_mu_loss', mu_loss)
        self.log(f'{phase}_sigma_loss', sigma_loss)
        self.log(f'{phase}_loss', loss)
        
        mu_adjusted_smape, mu_r2_score = self.get_score(mu_sigma[:,0], forecasted_mu_sigma[:,0])
        self.log(f'{phase}_mu_adjusted_smape', mu_adjusted_smape)
        self.log(f'{phase}_mu_r2_score', mu_r2_score)

        sigma_adjusted_smape, sigma_r2_score = self.get_score(mu_sigma[:,1], forecasted_mu_sigma[:,1])
        self.log(f'{phase}_sigma_adjusted_smape', sigma_adjusted_smape)
        self.log(f'{phase}_sigma_r2_score', sigma_r2_score)

        return loss