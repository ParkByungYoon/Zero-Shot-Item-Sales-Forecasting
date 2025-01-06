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

def inference(model, data_loader):
    device = next(model.parameters()).device
    dtw_prediction = []
    sorted_by_pred_dtw = []
    mse_losses = []

    hit_rates_at_10 = []
    hit_rates_at_50 = []
    hit_rates_at_100 = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(data_loader):
            dtw, center_items, neighbor_items = batch
            inf_idx = dtw.isinf().clone()
            dtw = dtw.to(device)
            center_items = [c[~inf_idx].to(device) for c in center_items]
            neighbor_items = [n[~inf_idx].to(device) for n in neighbor_items]

            pred = model(center_items, neighbor_items).squeeze()
            prediction = torch.full_like(dtw, float('inf'))
            prediction[~inf_idx] = pred

            y_true = torch.argsort(dtw).detach().cpu()
            y_pred = torch.argsort(prediction).detach().cpu()
            
            mse_loss = F.mse_loss(dtw, prediction)
            if ~torch.isnan(mse_loss):
                mse_losses.append(mse_loss.item())
            
            hit_rates_at_10.append(hit_rate(y_true, y_pred, 10))
            hit_rates_at_50.append(hit_rate(y_true, y_pred, 50))
            hit_rates_at_100.append(hit_rate(y_true, y_pred, 100))
            
            dtw_prediction.append(prediction.unsqueeze(0))
            sorted_by_pred_dtw.append(y_pred.unsqueeze(0))

    print('MSE:',np.mean(mse_losses))
    print('hit@10:',np.mean(hit_rates_at_10))
    print('hit@50:',np.mean(hit_rates_at_50))
    print('hit@100',np.mean(hit_rates_at_100))

    dtw_prediction = torch.cat(dtw_prediction, dim=0).detach().cpu()
    sorted_by_pred_dtw = torch.cat(sorted_by_pred_dtw, dim=0)

    return dtw_prediction, sorted_by_pred_dtw

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
    
    train_dataset, valid_dataset, test_dataset = [
        DTWSamplingDataset(
            dtw_matrix,
            total_ids,
            meta_df,
            release_date_df,
            image_embedding,
            text_embedding,
            num_samples=-1, 
            mode=f'{phase}'
        ) for phase in ['train_inf', 'valid_inf', 'test_inf']
    ]

    print(len(train_dataset), len(valid_dataset), len(test_dataset))
    print(len(train_dataset)//8251, len(valid_dataset)//2112, len(test_dataset)//1118)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

    model = DTWConcatenate( 
        embedding_dim=512,
        hidden_dim=512,
        lr=0.0001
    )
    model.load_state_dict(torch.load(os.path.join(args.ckpt_dir, f'{args.model_type}/{args.model_name}.ckpt'))['state_dict'], strict=False)
    model.to(f'cuda:{args.gpu_num}')

    train_dtw_prediction, train_sorted_by_pred_dtw = inference(model, train_loader)
    valid_dtw_prediction, valid_sorted_by_pred_dtw = inference(model, valid_loader)
    test_dtw_prediction, test_sorted_by_pred_dtw = inference(model, test_loader)

    dtw_prediction = torch.cat([train_dtw_prediction, valid_dtw_prediction, test_dtw_prediction], dim=0)
    sorted_by_pred_dtw = torch.cat([train_sorted_by_pred_dtw, valid_sorted_by_pred_dtw, test_sorted_by_pred_dtw], dim=0)

    np.save(os.path.join(args.data_dir, f'pred_dtw_{args.model_name}.npy'), np.array(dtw_prediction))
    np.save(os.path.join(args.data_dir, f'sorted_by_pred_dtw_{args.model_name}.npy'), np.int64(np.array(sorted_by_pred_dtw)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Zero-Shot-Item-Sales-Forecasting')
    # General arguments
    parser.add_argument('--data_dir', type=str, default='../data/preprocessed')
    parser.add_argument('--ckpt_dir', type=str, default='../log')
    parser.add_argument('--result_dir', type=str, default='../output')
    parser.add_argument('--seed', type=int, default=21)
    parser.add_argument('--gpu_num', type=int, default=2)

    # Model specific arguments
    parser.add_argument('--model_type', type=str, default='DTW-Predictor')
    parser.add_argument('--model_name', type=str, default='concatenate')
    parser.add_argument('--batch_size', type=int, default=8251)
    parser.add_argument('--embedding_dim', type=int, default=512)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--topk', type=int, default=100)

    # wandb arguments
    parser.add_argument('--wandb_entity', type=str, default='bonbak')
    parser.add_argument('--wandb_proj', type=str, default='Zero-Shot-Item-Sales-Forecasting-DTW')
    parser.add_argument('--wandb_dir', type=str, default='../')

    args = parser.parse_args()
    run(args)