import pandas as pd
import pickle
import os 
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
import numpy as np
from dateutil.relativedelta import relativedelta
from tqdm import tqdm


data_dir = '../data/raw'
save_dir = '../data/preprocessed'

def get_item_sales_per_week():
    sales_df = pd.read_csv(os.path.join(data_dir, 'item_sale_per_week_240807.csv')).set_index('item_number_color')

    sales_data = []
    release_dates = []

    for (idx, row) in tqdm(sales_df.iterrows(), total=len(sales_df), ascii=True):
        row.index = pd.to_datetime(row.index)
        release_date = row.dropna().sort_index().index[0]
        row = row[release_date:release_date + relativedelta(weeks=11)].resample('7d').sum().fillna(0)
        sales_data.append(row.values)
        release_dates.append(release_date)

    sales_df = pd.DataFrame(sales_data, index=sales_df.index)
    sales_df['release_date'] = release_dates
    sales_df.to_csv(os.path.join(save_dir, 'sales_week12.csv'))

    return sales_df


def train_test_split(sales_df):
    # Meta-Data
    meta_df = pd.read_csv(os.path.join(data_dir, 'meta_data_240331.csv')).set_index('item_number_color')
    meta_df.to_csv(os.path.join(save_dir, 'meta_data.csv'))
    meta_df = meta_df[meta_df.columns[meta_df.columns.str.startswith(('main_color', 'category', 'fabric'))]]

    # Image
    with open(os.path.join(save_dir, 'fclip_image_embedding.pkl'), 'rb') as f:
        image_embedding = pickle.load(f)

    # Text
    with open(os.path.join(save_dir, 'fclip_text_embedding.pickle'), 'rb') as f:
        text_embedding = pickle.load(f)

    # Train/Test Split
    train_idx = pickle.load(open(os.path.join(data_dir, "train_item_number.pkl"), 'rb'))
    test_idx = pickle.load(open(os.path.join(data_dir, "test_item_number.pkl"), 'rb'))

    train_df = sales_df.loc[(pd.Series([idx[:-2] for idx in sales_df.index]).isin(train_idx)).values]
    test_df = sales_df.loc[(pd.Series([idx[:-2] for idx in sales_df.index]).isin(test_idx)).values]

    train_idx = train_df.index.tolist()
    test_idx = test_df.index.tolist()

    train_idx = list(set(train_idx).intersection(image_embedding.keys()))
    test_idx = list(set(test_idx).intersection(image_embedding.keys()))

    train_idx = [item_id for item_id in train_idx if item_id[:-2] in text_embedding.keys()]
    test_idx = [item_id for item_id in test_idx if item_id[:-2] in text_embedding.keys()]

    sorted_item_ids = np.array(train_idx + test_idx)
    np.save(os.path.join(save_dir, 'idx4sort.npy'), sorted_item_ids)

    # Text + Image + Meta
    X_train = meta_df.loc[train_idx].values
    X_test = meta_df.loc[test_idx].values

    X_image_train = np.array([image_embedding[item_id] for item_id in train_idx])
    X_image_test = np.array([image_embedding[item_id] for item_id in test_idx])

    X_text_train = np.array([text_embedding[item_id[:-2]] for item_id in train_idx])
    X_text_test = np.array([text_embedding[item_id[:-2]] for item_id in test_idx])

    X_train = np.concatenate([X_train, X_image_train, X_text_train], axis=1)
    X_test = np.concatenate([X_test, X_image_test, X_text_test], axis=1)

    return X_train, X_test

def get_nearest_neighbors(X_train, X_test):
    # Euclidean Distance
    train_dist = euclidean_distances(X_train, X_train)
    np.fill_diagonal(train_dist, np.Inf)
    test_dist = euclidean_distances(X_test, X_train)
    total_dist = np.concatenate([train_dist, test_dist])
    dist_arg = np.argsort(total_dist, axis=1)
    np.save(os.path.join(save_dir, 'sorted_by_eucdist.npy'), dist_arg)

    # Consine Similarity
    train_sim = cosine_similarity(X_train, X_train)
    np.fill_diagonal(train_sim, -np.inf)
    test_sim = cosine_similarity(X_test, X_train)
    total_sim = np.concatenate([train_sim, test_sim])
    cossim_arg = np.argsort(total_sim, axis=1)[:, ::-1]
    np.save(os.path.join(save_dir, 'sorted_by_cossim.npy'), cossim_arg) 



def run():
    sales_df = get_item_sales_per_week()
    X_train, X_test = train_test_split(sales_df)
    get_nearest_neighbors(X_train, X_test)

if __name__ == '__main__':
    run()