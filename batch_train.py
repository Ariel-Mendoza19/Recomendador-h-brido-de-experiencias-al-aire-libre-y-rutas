import argparse
from utils import load_data, train_batch_svd, save_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_components", type=int, default=30)
    args = parser.parse_args()

    items, ratings = load_data()
    print("📦 Datos cargados:", items.shape, ratings.shape)

    print("⚙️ Entrenando modelo SVD (batch)...")
    model = train_batch_svd(ratings, n_components=args.n_components)
    save_model(model)
    print("✅ Modelo batch guardado en model/batch_svd.pkl")
