import sys
sys.path.append('../')
from model.waveform import *
from dataset import KNNZeroShotDataset
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
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

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

    sales_df = sales_df.iloc[:,:-1]
    idx4sort = np.load(os.path.join(args.data_dir, 'idx4sort.npy'))
    sorted_by_metric = np.load(os.path.join(args.data_dir, f'sorted_by_{args.knn_metric}.npy'))

    # Image
    with open(os.path.join(args.data_dir, 'image_embedding_fclip.pkl'), 'rb') as f:
        image_embedding = pickle.load(f)
    # Text
    with open(os.path.join(args.data_dir, 'text_embedding_fclip.pickle'), 'rb') as f:
        text_embedding = pickle.load(f)
    # Meta
    meta_df = pd.read_csv(os.path.join(args.data_dir, 'meta_data.csv')).set_index('item_number_color')
    meta_df = meta_df.iloc[:,4:]

    mu_sigma_df = pd.DataFrame()
    mu_sigma_df['sales_mean'], mu_sigma_df['sales_std'] = sales_df.mean(axis=1), sales_df.std(axis=1)

    train_dataset = KNNZeroShotDataset(
        sales_df,
        idx4sort,
        sorted_by_metric,
        meta_df,
        mu_sigma_df,
        release_date_df,
        image_embedding,
        text_embedding,
        num_neighbors=args.num_neighbors,
        train=True
    )

    test_dataset = KNNZeroShotDataset(
        sales_df,
        idx4sort,
        sorted_by_metric,
        meta_df,
        mu_sigma_df,
        release_date_df,
        image_embedding,
        text_embedding,
        num_neighbors=args.num_neighbors,
        train=False
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, num_workers=8)

    model = KNNTransformerWaveform(
        input_len=args.input_len,
        output_len=args.output_len,
        num_neighbors=args.num_neighbors,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        lr=0.0001
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=f'{args.log_dir}/{args.model_type}',
        filename=f'K{args.num_neighbors}-{args.knn_metric}',
        monitor='valid_adjusted_smape',
        mode='min',
        save_top_k=1
    )

    wandb.require("core")
    wandb.init(
        entity=args.wandb_entity, 
        project=args.wandb_proj, 
        name=f'K{args.num_neighbors}-{args.knn_metric}',
        dir=args.wandb_dir
    )
    wandb_logger = pl_loggers.WandbLogger()
    trainer = pl.Trainer(
        devices=[args.gpu_num],
        max_epochs=args.num_epochs,
        check_val_every_n_epoch=1,
        logger=wandb_logger,
        callbacks=[checkpoint_callback]
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader)

    print(checkpoint_callback.best_model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Zero-Shot-Item-Sales-Forecasting')
    # General arguments
    parser.add_argument('--data_dir', type=str, default='../data/preprocessed')
    parser.add_argument('--log_dir', type=str, default='../log')
    parser.add_argument('--seed', type=int, default=21)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--gpu_num', type=int, default=1)

    # Model specific arguments
    parser.add_argument('--model_type', type=str, default='KNN-Transformer')
    parser.add_argument('--knn_metric', type=str, default='eucdist')
    parser.add_argument('--num_neighbors', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--embedding_dim', type=int, default=512)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--input_len', type=int, default=12)
    parser.add_argument('--output_len', type=int, default=12)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=2)

    # wandb arguments
    parser.add_argument('--wandb_entity', type=str, default='bonbak')
    parser.add_argument('--wandb_proj', type=str, default='Zero-Shot-Item-Sales-Forecasting-Waveform')
    parser.add_argument('--wandb_dir', type=str, default='../')

    args = parser.parse_args()
    run(args)