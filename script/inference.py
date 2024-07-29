import os 
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from model.waveform import *
from dataset import ZeroShotDataset
from torch.utils.data import DataLoader
import torch
import pandas as pd
import pickle
import argparse
import numpy as np
from tqdm import tqdm
import random
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def run(args):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    sales_df = pd.read_csv(os.path.join(args.data_dir, 'sales_week12.csv')).set_index('item_number_color')
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

    train_dataset = ZeroShotDataset(
        sales_df,
        idx4sort,
        sorted_by_metric,
        meta_df,
        mu_sigma_df,
        image_embedding,
        text_embedding,
        k=args.num_neighbors,
        train=True
    )

    test_dataset = ZeroShotDataset(
        sales_df,
        idx4sort,
        sorted_by_metric,
        meta_df,
        mu_sigma_df,
        image_embedding,
        text_embedding,
        k=args.num_neighbors,
        train=False
    )

    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False, num_workers=8)
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
    model.load_state_dict(torch.load(os.path.join(args.ckpt_dir, f'K{args.num_neighbors}-{args.knn_metric}.ckpt'))['state_dict'], strict=False)

    model.eval()
    with torch.no_grad():
        for batch in train_loader:
            item_sales, _, k_item_sales, _, image_embeddings, text_embeddings, meta_data = batch
            forecasted_sales_train, _ = model(k_item_sales, image_embeddings, text_embeddings, meta_data)
            adjusted_smape, r2_score = model.get_score(item_sales, forecasted_sales_train)
        print('train:',adjusted_smape, r2_score)

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            item_sales, _, k_item_sales, _, image_embeddings, text_embeddings, meta_data = batch
            forecasted_sales_test, _ = model(k_item_sales, image_embeddings, text_embeddings, meta_data)
            adjusted_smape, r2_score = model.get_score(item_sales, forecasted_sales_test)
        print('test:',adjusted_smape, r2_score)

        forecasted_sales = torch.cat([forecasted_sales_train, forecasted_sales_test], dim=0)
    df_index = train_dataset.item_ids.tolist()+test_dataset.item_ids.tolist()
    df = pd.DataFrame(index=df_index, data=forecasted_sales)
    df.to_csv(os.path.join(args.result_dir, f'result_K{args.num_neighbors}-{args.knn_metric}.csv'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Zero-Shot-Item-Sales-Forecasting')
    # General arguments
    parser.add_argument('--data_dir', type=str, default='../data/preprocessed')
    parser.add_argument('--ckpt_dir', type=str, default='../log/KNN-Transformer')
    parser.add_argument('--result_dir', type=str, default='../output')
    parser.add_argument('--seed', type=int, default=21)
    parser.add_argument('--gpu_num', type=int, default=1)

    # Model specific arguments
    parser.add_argument('--model_type', type=str, default='KNN-Transformer')
    parser.add_argument('--knn_metric', type=str, default='eucdist')
    parser.add_argument('--num_neighbors', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--embedding_dim', type=int, default=512)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--input_len', type=int, default=12)
    parser.add_argument('--output_len', type=int, default=12)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--num_layers', type=int, default=2)

    args = parser.parse_args()
    run(args)