import pandas as pd
import numpy as np
from sklearn.decomposition import TruncatedSVD
import joblib
from pathlib import Path

DATA_DIR = Path("data")
MODEL_DIR = Path("model")
MODEL_DIR.mkdir(exist_ok=True)

def load_data():
    items = pd.read_csv(DATA_DIR / "items.csv")
    ratings = pd.read_csv(DATA_DIR / "sample_ratings.csv")
    return items, ratings

def create_user_item_matrix(ratings_df):
    pivot = ratings_df.pivot_table(index="user_id", columns="item_id", values="rating", fill_value=0)
    return pivot

def train_batch_svd(ratings_df, n_components=50):
    pivot = create_user_item_matrix(ratings_df)
    svd = TruncatedSVD(n_components=min(n_components, min(pivot.shape)-1), random_state=42)
    U = svd.fit_transform(pivot.values)
    Sigma = svd.singular_values_
    Vt = svd.components_
    user_index = list(pivot.index)
    item_index = list(pivot.columns)
    return {"svd": svd, "U": U, "Sigma": Sigma, "Vt": Vt, "user_index": user_index, "item_index": item_index}

def save_model(obj, name="batch_svd.pkl"):
    joblib.dump(obj, MODEL_DIR / name)

def load_model(name="batch_svd.pkl"):
    return joblib.load(MODEL_DIR / name)

def batch_recommend_for_user(batch_model, user_id, top_k=10):
    if user_id not in batch_model["user_index"]:
        return []
    idx = batch_model["user_index"].index(user_id)
    user_vec = batch_model["U"][idx]
    item_vecs = batch_model["Vt"].T
    scores = item_vecs.dot(user_vec)
    item_ids = batch_model["item_index"]
    ranked = sorted(zip(item_ids, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]
