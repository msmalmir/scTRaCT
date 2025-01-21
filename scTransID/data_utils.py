import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import scanpy as sc

def load_data(train_path, test_path):
    train_adata = sc.read_h5ad(train_path)
    query_adata = sc.read_h5ad(test_path)
    return train_adata, query_adata

def preprocess_data(train_adata, query_adata, cell_type_key):
    le = LabelEncoder()
    y_train = le.fit_transform(train_adata.obs[cell_type_key])
    X_train = train_adata.X.A if hasattr(train_adata.X, "A") else train_adata.X
    X_query = query_adata.X.A if hasattr(query_adata.X, "A") else query_adata.X
    return X_train, y_train, X_query, le

def split_data(X, y):
    X_train_split, X_val, y_train_split, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    return (torch.tensor(X_train_split, dtype=torch.float32),
            torch.tensor(y_train_split, dtype=torch.long),
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long))
