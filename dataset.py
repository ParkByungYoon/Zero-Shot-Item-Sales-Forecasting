import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset

class KNNZeroShotDataset(Dataset):
    def __init__(self,
                 sales_df,
                 sorted_item_ids,
                 sorted_by_metric,
                 meta_df,
                 mu_sigma_df,
                 release_date_df,
                 image_embedding,
                 text_embedding,
                 num_neighbors=10, 
                 train=True):
        
        self.sales_df = sales_df
        self.sorted_item_ids = sorted_item_ids
        self.sorted_by_metric = sorted_by_metric
        self.meta_df = meta_df
        self.mu_sigma_df = mu_sigma_df
        self.release_date_df = release_date_df
        self.image_embedding = image_embedding
        self.text_embedding = text_embedding
        self.num_neighbors = num_neighbors
        self.item_ids = self.sorted_item_ids[:-2118] if train else self.sorted_item_ids[-2118:]

        self.__preprocess__()
    
    def __preprocess__(self):
        
        item_sales, mu_and_sigma, k_item_sales, k_mu_and_sigma, release_dates, image_embeddings, text_embeddings, meta_data = [[] for _ in range(8)]


        for item_id in tqdm(self.item_ids, total=len(self.item_ids), ascii=True):
            mu_sigma = self.mu_sigma_df.loc[item_id].values
            sales = ((self.sales_df.loc[item_id]-mu_sigma[0]) / mu_sigma[1]).values            
            
            target_idx = np.where(self.sorted_item_ids == item_id)[0][0]
            k_nearest_idx = self.sorted_by_metric[target_idx][:self.num_neighbors]
            k_nearest_item_ids = self.sorted_item_ids[k_nearest_idx]

            k_mu_sigma = self.mu_sigma_df.loc[k_nearest_item_ids].values
            k_sales = ((self.sales_df.loc[k_nearest_item_ids].values - k_mu_sigma[:,0:1]) / k_mu_sigma[:,1:])
            
            release_date = self.release_date_df.loc[item_id].values
            img_emb = self.image_embedding[item_id]
            txt_emb = self.text_embedding[item_id[:-2]]
            meta = self.meta_df.loc[item_id].values

            item_sales.append(sales)
            mu_and_sigma.append(mu_sigma)
            k_item_sales.append(k_sales)
            k_mu_and_sigma.append(k_mu_sigma)
            release_dates.append(release_date)
            image_embeddings.append(img_emb)
            text_embeddings.append(txt_emb)
            meta_data.append(meta)
        
        self.item_sales = torch.FloatTensor(item_sales)
        self.mu_sigma = torch.FloatTensor(mu_and_sigma)
        self.k_item_sales = torch.FloatTensor(k_item_sales)
        self.release_dates = torch.FloatTensor(release_dates)
        self.k_mu_sigma = torch.FloatTensor(k_mu_and_sigma)
        self.image_embeddings = torch.FloatTensor(image_embeddings)
        self.text_embeddings = torch.FloatTensor(text_embeddings)
        self.meta_data = torch.FloatTensor(meta_data)

    def __getitem__(self, idx):
        return \
            self.item_sales[idx],\
            self.mu_sigma[idx],\
            self.k_item_sales[idx], \
            self.k_mu_sigma[idx],\
            self.release_dates[idx],\
            self.image_embeddings[idx],\
            self.text_embeddings[idx],\
            self.meta_data[idx]
    
    def __len__(self):
        return len(self.item_ids)
    


class NTrendZeroShotDataset(Dataset):
    def __init__(self,
                 sales_df,
                 sorted_item_ids,
                 category_df,
                 color_df,
                 fabric_df,
                 meta_df,
                 mu_sigma_df,
                 release_date_df,
                 image_embedding,
                 text_embedding, 
                 train=True):

        self.sales_df = sales_df
        self.sorted_item_ids = sorted_item_ids
        self.category_df = category_df
        self.color_df = color_df
        self.fabric_df = fabric_df
        self.meta_df = meta_df
        self.mu_sigma_df = mu_sigma_df
        self.release_date_df = release_date_df
        self.image_embedding = image_embedding
        self.text_embedding = text_embedding
        self.item_ids = self.sorted_item_ids[:-2118] if train else self.sorted_item_ids[-2118:]

        self.__preprocess__()
    
    def __preprocess__(self):
        item_sales, mu_and_sigma, ntrends, release_dates, image_embeddings, text_embeddings, meta_data = [[] for _ in range(7)]

        for item_id in tqdm(self.item_ids, total=len(self.item_ids), ascii=True):
            mu_sigma = self.mu_sigma_df.loc[item_id].values
            sales = ((self.sales_df.loc[item_id]-mu_sigma[0]) / mu_sigma[1]).values

            category_trend = self.category_df.loc[item_id].values
            color_trend = self.color_df.loc[item_id].values
            fabric_trend = self.fabric_df.loc[item_id].values
            ntrend = np.vstack([category_trend, color_trend, fabric_trend])
            
            release_date = self.release_date_df.loc[item_id].values
            img_emb = self.image_embedding[item_id]
            txt_emb = self.text_embedding[item_id[:-2]]
            meta = self.meta_df.loc[item_id].values

            item_sales.append(sales)
            mu_and_sigma.append(mu_sigma)
            ntrends.append(ntrend)
            release_dates.append(release_date)
            image_embeddings.append(img_emb)
            text_embeddings.append(txt_emb)
            meta_data.append(meta)
        
        self.item_sales = torch.FloatTensor(item_sales)
        self.mu_sigma = torch.FloatTensor(mu_and_sigma)
        self.ntrends = torch.FloatTensor(ntrends)
        self.release_dates = torch.FloatTensor(release_dates)
        self.image_embeddings = torch.FloatTensor(image_embeddings)
        self.text_embeddings = torch.FloatTensor(text_embeddings)
        self.meta_data = torch.FloatTensor(meta_data)

    def __getitem__(self, idx):
        return \
            self.item_sales[idx],\
            self.mu_sigma[idx],\
            self.ntrends[idx],\
            self.ntrends[idx],\
            self.release_dates[idx],\
            self.image_embeddings[idx],\
            self.text_embeddings[idx],\
            self.meta_data[idx]
    
    def __len__(self):
        return len(self.item_ids)
    

class DTWSamplingDataset(Dataset):
    def __init__(self,
                 dtw_matrix,
                 sorted_item_ids,
                 meta_df,
                 release_date_df,
                 image_embedding,
                 text_embedding,
                 num_neighbors=10, 
                 train=True):    
        
        self.sorted_item_ids = sorted_item_ids
        self.meta_df = meta_df
        self.release_date_df = release_date_df
        self.image_embedding = image_embedding
        self.text_embedding = text_embedding
        self.num_neighbors = num_neighbors
        self.item_ids = self.sorted_item_ids[:-2118] if train else self.sorted_item_ids[-2118:]
        self.dtw_matrix = dtw_matrix[0] if train else dtw_matrix[-2118:,0].squeeze()

        self.__preprocess__()
    
    def __preprocess__(self):
        
        release_dates, image_embeddings, text_embeddings, meta_data = [[] for _ in range(4)]

        for item_id in tqdm(self.item_ids, total=len(self.item_ids), ascii=True):
            release_date = self.release_date_df.loc[item_id].values
            img_emb = self.image_embedding[item_id]
            txt_emb = self.text_embedding[item_id[:-2]]
            meta = self.meta_df.loc[item_id].values
            
            release_dates.append(release_date)
            image_embeddings.append(img_emb)
            text_embeddings.append(txt_emb)
            meta_data.append(meta)
        
        self.release_dates = torch.FloatTensor(release_dates)
        self.image_embeddings = torch.FloatTensor(image_embeddings)
        self.text_embeddings = torch.FloatTensor(text_embeddings)
        self.meta_data = torch.FloatTensor(meta_data)

    def __getitem__(self, idx):
        center_item = \
            self.image_embeddings[0],\
            self.text_embeddings[0],\
            self.meta_data[0],\
            self.release_dates[0],
    
        neighbor_item = \
            self.image_embeddings[idx+1],\
            self.text_embeddings[idx+1],\
            self.meta_data[idx+1],\
            self.release_dates[idx+1]
            
        dtw = self.dtw_matrix[idx+1]
        return dtw, center_item, neighbor_item
    
    def __len__(self):
        return len(self.item_ids) - 1