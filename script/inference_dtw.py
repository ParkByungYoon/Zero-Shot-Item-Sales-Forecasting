import sys
sys.path.append('../')
from model.dtw import *
from dataset import DTWSamplingDataset
from torch.utils.data import DataLoader
import torch
import pandas as pd
import pickle
import argparse
import os 
import numpy as np
import wandb
import random

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from tqdm import tqdm
import urllib3

torch.autograd.set_detect_anomaly(True)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def hit_rate(y_true, y_pred, k):
    return len(np.intersect1d(y_true[:k], y_pred[:k])) / k

def run(args):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    sales_df = pd.read_csv(os.path.join(args.data_dir, 'sales_week12.csv')).set_index('item_number_color')

    release_date_df = pd.DataFrame(index=sales_df.index)
    release_date = pd.to_datetime(sales_df['release_date'])
    release_date_df['year']= release_date.apply(lambda x:x.year)
    release_date_df['month']= release_date.apply(lambda x:x.month)
    release_date_df['week']= release_date.apply(lambda x:x.week)
    release_date_df['day']= release_date.apply(lambda x:x.day)

    total_ids = np.load(os.path.join(args.data_dir, 'total_ids.npy'))
    dtw_matrix = np.load(os.path.join(args.data_dir, f'total_dtw_dist.npy'))

    # Image
    with open(os.path.join(args.data_dir, 'fclip_image_embedding.pickle'), 'rb') as f:
        image_embedding = pickle.load(f)
    # Text
    with open(os.path.join(args.data_dir, 'fclip_text_embedding.pickle'), 'rb') as f:
        text_embedding = pickle.load(f)
    # Meta
    meta_df = pd.read_csv(os.path.join(args.data_dir, 'meta_data.csv')).set_index('item_number_color')
    
    test_dataset = DTWSamplingDataset(
        dtw_matrix[-1118:],
        total_ids,
        meta_df,
        release_date_df,
        image_embedding,
        text_embedding,
        num_samples=-1, 
        mode='test'
    ) 

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    model = DTWDotProduct( 
        embedding_dim=512,
        hidden_dim=512,
        lr=0.0001
    )
    model.load_state_dict(torch.load(os.path.join(args.ckpt_dir, f'{args.model_type}/dtw-dot-product.ckpt'))['state_dict'], strict=False)

    model.eval()
    test_losses = []
    hit_rates_at_10 = []
    hit_rates_at_50 = []
    hit_rates_at_100 = []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            dtw, center_items, neighbor_items = batch
            prediction = model(center_items, neighbor_items)
            test_loss = F.mse_loss(dtw.squeeze(), prediction.squeeze())
            test_losses.append(test_loss.item())
            
            y_true = torch.topk(dtw, args.topk, largest=False).indices
            y_pred = torch.topk(prediction.squeeze(), args.topk, largest=False).indices

            hit_rates_at_10.append(hit_rate(y_true, y_pred, 10))
            hit_rates_at_50.append(hit_rate(y_true, y_pred, 50))
            hit_rates_at_100.append(hit_rate(y_true, y_pred, 100))

    print('MSE:',np.mean(test_losses))
    print('hit@10:',np.mean(hit_rates_at_10))
    print('hit@50:',np.mean(hit_rates_at_50))
    print('hit@100',np.mean(hit_rates_at_100))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Zero-Shot-Item-Sales-Forecasting')
    # General arguments
    parser.add_argument('--data_dir', type=str, default='../data/preprocessed')
    parser.add_argument('--ckpt_dir', type=str, default='../log')
    parser.add_argument('--result_dir', type=str, default='../output')
    parser.add_argument('--seed', type=int, default=21)
    parser.add_argument('--gpu_num', type=int, default=1)

    # Model specific arguments
    parser.add_argument('--model_type', type=str, default='DTW-Predictor')
    parser.add_argument('--batch_size', type=int, default=8250)
    parser.add_argument('--embedding_dim', type=int, default=512)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--topk', type=int, default=100)

    # wandb arguments
    parser.add_argument('--wandb_entity', type=str, default='bonbak')
    parser.add_argument('--wandb_proj', type=str, default='Zero-Shot-Item-Sales-Forecasting-DTW')
    parser.add_argument('--wandb_dir', type=str, default='../')

    args = parser.parse_args()
    run(args)