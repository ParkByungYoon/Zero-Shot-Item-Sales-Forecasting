from model import KNNTransformer
from dataset import ZeroShotDataset
from torch.utils.data import DataLoader
import torch
import pandas as pd
import pickle
import argparse
import os 
import numpy as np
import wandb

from datetime import datetime
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

def run(args):
    print(args)
    torch.manual_seed(args.seed)

    sales_df = pd.read_csv(os.path.join(args.data_dir, 'sales_week12.csv')).set_index('item_number_color')
    sorted_item_ids = np.load(os.path.join(args.data_dir, 'sorted_item_ids.npy'))
    sorted_by_eucdist = np.load(os.path.join(args.data_dir, 'sorted_by_eucdist.npy'))

    # Image
    with open(os.path.join(args.data_dir, 'image_embedding_fclip.pkl'), 'rb') as f:
        image_embedding = pickle.load(f)
    # Text
    with open(os.path.join(args.data_dir, 'text_embedding_fclip.pickle'), 'rb') as f:
        text_embedding = pickle.load(f)
    # Meta
    meta_df = pd.read_csv(os.path.join(args.data_dir, 'meta_data.csv')).set_index('item_number_color')
    scale_df = meta_df[['sales_mean', 'sales_std']]
    meta_df = meta_df.iloc[:,4:]

    train_dataset = ZeroShotDataset(
        sales_df,
        sorted_item_ids,
        sorted_by_eucdist,
        meta_df,
        scale_df,
        image_embedding,
        text_embedding,
        k=args.num_trends,
        train=True
    )

    test_dataset = ZeroShotDataset(
        sales_df,
        sorted_item_ids,
        sorted_by_eucdist,
        meta_df,
        scale_df,
        image_embedding,
        text_embedding,
        k=args.num_trends,
        train=False
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(train_dataset, batch_size=len(test_dataset), shuffle=False, num_workers=4)

    model = KNNTransformer(
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        trend_len=args.output_dim,
        num_trends=args.num_trends,
        gpu_num=args.gpu_num,
        lr=0.0001
    )


    # Model Training
    # Define model saving procedure
    dt_string = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=f'{args.log_dir}/{args.model_type}',
        filename=args.model_type+'---{epoch}---'+dt_string,
        monitor='valid_adjusted_smape',
        mode='min',
        save_top_k=1
    )

    wandb.require("core")
    wandb.init(entity=args.wandb_entity, project=args.wandb_proj, name=args.wandb_run)
    wandb_logger = pl_loggers.WandbLogger()
    trainer = pl.Trainer(
        devices=[args.gpu_num],
        max_epochs=args.num_epochs,
        check_val_every_n_epoch=5,
        logger=wandb_logger,
        callbacks=[checkpoint_callback]
    )
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=test_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Zero-shot sales forecasting')
    # General arguments
    parser.add_argument('--data_dir', type=str, default='data/preprocessed')
    parser.add_argument('--log_dir', type=str, default='log')
    parser.add_argument('--seed', type=int, default=21)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--gpu_num', type=int, default=1)

    # Model specific arguments
    parser.add_argument('--model_type', type=str, default='KNN-Transformer')
    parser.add_argument('--trend_len', type=int, default=12)
    parser.add_argument('--num_trends', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--embedding_dim', type=int, default=512)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--output_dim', type=int, default=12)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=2)

    # wandb arguments
    parser.add_argument('--wandb_entity', type=str, default='bonbak')
    parser.add_argument('--wandb_proj', type=str, default='Zero-Shot Item Sales Forecasting')
    parser.add_argument('--wandb_run', type=str, default='Run1')

    args = parser.parse_args()
    run(args)