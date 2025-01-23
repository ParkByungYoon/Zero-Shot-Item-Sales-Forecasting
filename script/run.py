import sys
sys.path.append('../')

from model.gtm import *
import torch
import argparse
import numpy as np
import wandb
import random
from util.datamodule import MultiVariateDataModule
import datetime

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import os


model_dict = {
    "GTM-Transformer": GTMTransformer,
    "GTM-iTransformer": GTMiTransformer,
    "GTM-Crossformer": GTMCrossformer,
    "GTM-Fullformer": GTMFullformer,
}

def random_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed) 
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False 
    torch.autograd.set_detect_anomaly(True)

def run(args):
    print(args)
    random_seed(args.seed)


    model = model_dict[args.model_type](args)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=os.path.join(args.log_dir,args.model_type),
        filename=f'{args.model_type}-{datetime.datetime.now().strftime("%y%m%d-%H%M")}',
        monitor='valid_adjusted_smape',
        mode='min',
        save_top_k=1
    )

    wandb.require("core")
    wandb.init(
        entity=args.wandb_entity, 
        project=args.wandb_proj, 
        name=f'{args.model_type}-{datetime.datetime.now().strftime("%y%m%d-%H%M")}',
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
    trainer.fit(model, datamodule=MultiVariateDataModule(args))
    print(checkpoint_callback.best_model_path)
    ckpt_path = checkpoint_callback.best_model_path
    trainer.test(model=model, ckpt_path=ckpt_path, datamodule=MultiVariateDataModule(args))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Zero-Shot-Item-Sales-Forecasting')
    # General arguments
    parser.add_argument('--data_dir', type=str, default='../data/lightning')
    parser.add_argument('--log_dir', type=str, default='../log')
    parser.add_argument('--seed', type=int, default=21)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--gpu_num', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.0001)

    # Model specific arguments
    parser.add_argument('--model_type', type=str, default='GTM-Transformer')
    parser.add_argument('--num_vars', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--embedding_dim', type=int, default=512)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--input_len', type=int, default=52)
    parser.add_argument('--output_len', type=int, default=12)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--segment_len', type=int, default=4)

    # wandb arguments
    parser.add_argument('--wandb_entity', type=str, default='bonbak')
    parser.add_argument('--wandb_proj', type=str, default='Zero-Shot-Item-Sales-Forecasting')
    parser.add_argument('--wandb_dir', type=str, default='../')

    args = parser.parse_args()
    run(args)