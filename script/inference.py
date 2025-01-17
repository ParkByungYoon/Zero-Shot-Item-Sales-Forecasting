import sys
sys.path.append('../')

from model.gtm import *
import torch
import argparse
import numpy as np
import random
from util.datamodule import ZeroShotDataModule

import pytorch_lightning as pl
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import os
import pandas as pd


model_dict = {
    "GTM-Transformer": GTMTransformer,
    "GTM-iTransformer": GTMiTransformer,
    "GTMCrossFormer": GTMCrossformer,
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
    data = ZeroShotDataModule(args)
    trainer = pl.Trainer(
        devices=[args.gpu_num],
        logger=False,
    )
    ckpt_path = os.path.join(args.log_dir,args.model_type, f"{args.model_type}-{args.ckpt_name}")
    prediction = trainer.predict(model=model, ckpt_path=ckpt_path, datamodule=data)
    
    result_df = pd.DataFrame(torch.cat(prediction, dim=0).numpy(), index=data.test_dataset.item_ids)
    result_df.to_csv(os.path.join(args.result_dir, f"{args.model_type}-{args.ckpt_name}".replace("ckpt","csv")))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Zero-Shot-Item-Sales-Forecasting')
    # General arguments
    parser.add_argument('--data_dir', type=str, default='../data/lightning')
    parser.add_argument('--log_dir', type=str, default='../log')
    parser.add_argument('--result_dir', type=str, default='../result')
    parser.add_argument('--seed', type=int, default=21)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--gpu_num', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--ckpt_name', type=str, default="")

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

    args = parser.parse_args()
    run(args)