import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.regression import R2Score, SymmetricMeanAbsolutePercentageError

from layer import *
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class KNNTransformer(pl.LightningModule):
    def __init__(self, embedding_dim, hidden_dim, output_len, num_heads, num_layers, input_len, num_items, gpu_num, lr):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.output_len = output_len
        self.gpu_num = gpu_num
        self.save_hyperparameters()
        self.lr = lr

        self.item_sales_encoder = ItemSalesEncoder(hidden_dim, input_len, num_items, gpu_num)
        self.static_feature_encoder = StaticFeatureEncoder(embedding_dim, hidden_dim)

        # Decoder
        self.decoder_linear = TimeDistributed(nn.Linear(1, hidden_dim))
        decoder_layer = TransformerDecoderLayer(d_model=self.hidden_dim, nhead=num_heads, \
                                                dim_feedforward=self.hidden_dim * 4, dropout=0.1)
        
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        
        self.decoder_fc = nn.Sequential(
            nn.Linear(hidden_dim, self.output_len),
            nn.Dropout(0.2)
        )

    def _generate_square_subsequent_mask(self, size):
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0)).to('cuda:'+str(self.gpu_num))
        return mask

    def forward(self, k_item_sales, image_embedding, text_embedding, meta_data):
        # Encode features and get inputs
        k_item_sales_embedding = self.item_sales_encoder(k_item_sales)

        # Fuse static features together
        static_feature_fusion = self.static_feature_encoder(image_embedding, text_embedding, meta_data)
            
        tgt = static_feature_fusion.unsqueeze(0)
        memory = k_item_sales_embedding
        decoder_out, attn_weights = self.decoder(tgt, memory)
        forecast = self.decoder_fc(decoder_out)

        return forecast.view(-1, self.output_len), attn_weights

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def training_step(self, train_batch, batch_idx):
        output = self.phase_step(train_batch, phase='train')
        return output[0]

    def validation_step(self, valid_batch, batch_idx):
        self.phase_step(valid_batch, phase='valid')

    def test_step(self, test_batch, batch_idx):
        self.phase_step(test_batch, phase='test')

    def phase_step(self, batch, phase):
        item_sales, scale_factors, k_item_sales, image_embeddings, text_embeddings, meta_data = batch
        forecasted_sales, _ = self.forward(k_item_sales, image_embeddings, text_embeddings, meta_data)
        
        loss = F.mse_loss(item_sales, forecasted_sales.squeeze())
        adjusted_smape, r2_score = self.get_score(item_sales, forecasted_sales)
    
        self.log(f'{phase}_loss', loss)
        self.log(f'{phase}_adjusted_smape', adjusted_smape)
        self.log(f'{phase}_r2_score', r2_score)

        # return loss, adjusted_smape, r2_score

        rescaled_item_sales = item_sales * scale_factors[:,1].unsqueeze(dim=-1) + scale_factors[:,0].unsqueeze(dim=-1)
        rescaled_forecasted_sales = forecasted_sales * scale_factors[:,1].unsqueeze(dim=-1) + scale_factors[:,0].unsqueeze(dim=-1)
        rescaled_adjusted_smape, rescaled_r2_score = self.get_score(rescaled_item_sales, rescaled_forecasted_sales)

        self.log(f'{phase}_rescaled_adjusted_smape', rescaled_adjusted_smape)
        self.log(f'{phase}_rescaled_r2_score', rescaled_r2_score)

        return loss, adjusted_smape, r2_score, rescaled_adjusted_smape, rescaled_r2_score

    def get_score(self, gt, pred):
        ad_smape = SymmetricMeanAbsolutePercentageError()
        r2 = R2Score()
        pred = pred.detach().cpu()
        gt = gt.detach().cpu()
        
        adjust_smape = [ad_smape(pred[i], gt[i]) * 0.5 for i in range(len(gt))]
        adjust_smape = torch.mean(torch.stack(adjust_smape))

        r2_score = [r2(pred[i], gt[i]) for i in range(len(gt))]
        r2_score = torch.mean(torch.stack(r2_score))
        return adjust_smape, r2_score