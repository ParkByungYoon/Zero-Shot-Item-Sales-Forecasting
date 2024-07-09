import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset

class ZeroShotDataset(Dataset):
    def __init__(self,
                 sales_df,
                 sorted_item_ids,
                 sorted_by_eucdist,
                 meta_df,
                 scale_df,
                 image_embedding,
                 text_embedding,
                 k=10, 
                 train=True):
        
        self.k = k
        self.sales_df = sales_df
        self.sorted_item_ids = sorted_item_ids
        self.sorted_by_eucdist = sorted_by_eucdist
        self.meta_df = meta_df
        self.scale_df = scale_df
        self.image_embedding = image_embedding
        self.text_embedding = text_embedding
        self.item_ids = self.sorted_item_ids[:-2118] if train else self.sorted_item_ids[-2118:]

        self.__preprocess__()
    
    def __preprocess__(self):
        
        item_sales, scale_factors, nearest_neighbor_sales, image_embeddings, text_embeddings, meta_data = [[] for _ in range(6)]


        for item_id in tqdm(self.item_ids, total=len(self.item_ids), ascii=True):
            scale = self.scale_df.loc[item_id].values
            sales = ((self.sales_df.loc[item_id]-scale[0]) / scale[1]).values            
            
            target_idx = np.where(self.sorted_item_ids == item_id)[0][0] # distance matrix 에서의 index
            k_nearest_idx = self.sorted_by_eucdist[target_idx][:self.k]
            k_nearest_item_ids = self.sorted_item_ids[k_nearest_idx]
            nn_sales = self.sales_df.loc[k_nearest_item_ids].values
            nn_sales = (nn_sales - nn_sales.mean()) / nn_sales.std()
            
            img_emb = self.image_embedding[item_id]
            txt_emb = self.text_embedding[item_id[:-2]]
            meta = self.meta_df.loc[item_id].values

            nearest_neighbor_sales.append(nn_sales)
            scale_factors.append(scale)
            item_sales.append(sales)
            image_embeddings.append(img_emb)
            text_embeddings.append(txt_emb)
            meta_data.append(meta)
        
        self.item_sales = torch.FloatTensor(item_sales)
        self.nearest_neighbor_sales = torch.FloatTensor(nearest_neighbor_sales)
        self.scale_factors = torch.FloatTensor(scale_factors)
        self.image_embeddings = torch.FloatTensor(image_embeddings)
        self.text_embeddings = torch.FloatTensor(text_embeddings)
        self.meta_data = torch.FloatTensor(meta_data)

    def __getitem__(self, idx):
        return \
            self.item_sales[idx],\
            self.scale_factors[idx],\
            self.nearest_neighbor_sales[idx],\
            self.image_embeddings[idx],\
            self.text_embeddings[idx],\
            self.meta_data[idx]
    
    def __len__(self):
        return len(self.item_ids)