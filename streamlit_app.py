import streamlit as st
import pandas as pd
import numpy as np
from utils import load_data, load_model, batch_recommend_for_user
from pathlib import Path

st.set_page_config(page_title="Lambda Recommender - Outdoors", layout="centered")

MODEL_PATH = Path("model/batch_svd.pkl")
RECENT_FILE = Path("data/recent_ratings.csv")

st.title("🏞️ Recomendador Híbrido (Arquitectura Lambda) — Experiencias al aire libre")

# Cargar datos
items, ratings = load_data()
if MODEL_PATH.exists():
    model = load_model()
    st.success("Modelo batch cargado correctamente.")
else:
    st.warning("No hay modelo batch entrenado. Ejecuta: python batch_train.py")
    model = None

# Sidebar
st.sidebar.header("Configuración")
alpha = st.sidebar.slider("Peso de la capa Speed (α)", 0.0, 1.0, 0.35, 0.05)
top_k = st.sidebar.number_input("Cantidad de recomendaciones", 3, 20, 8)

# Usuario
st.subheader("Simular usuario")
user_id = st.text_input("User ID", value="user1")

# Speed Layer
if RECENT_FILE.exists():
    recent = pd.read_csv(RECENT_FILE)
else:
    recent = pd.DataFrame(columns=["user_id","item_id","rating","timestamp"])

st.subheader("Últimas valoraciones (Speed Layer)")
st.table(recent.sort_values("timestamp", ascending=False).head(10))

# Nueva valoración
st.subheader("Agregar nueva valoración (simulación en tiempo real)")
col1, col2, col3 = st.columns([2,2,1])
with col1:
    new_item = st.selectbox("Selecciona una experiencia", items["item_id"].tolist())
with col2:
    new_rating = st.slider("Valoración", 1, 5, 4)
with col3:
    if st.button("Enviar"):
        from speed_ingest import append_rating
        append_rating(user_id, new_item, new_rating)
        st.experimental_rerun()

# Generar recomendaciones
st.subheader("Recomendaciones híbridas")

batch_scores = {}
if model and user_id in model["user_index"]:
    for it, s in batch_recommend_for_user(model, user_id, top_k=50):
        batch_scores[it] = float(s)
else:
    avg = ratings.groupby("item_id")["rating"].mean().to_dict()
    for it in items["item_id"]:
        batch_scores[it] = avg.get(it, 3.0)

recent_user = recent[recent["user_id"] == user_id]
speed_scores = {}
if not recent_user.empty:
    for _, row in recent_user.iterrows():
        speed_scores[row["item_id"]] = row["rating"]
else:
    top_recent = recent.sort_values("timestamp", ascending=False).head(50)
    item_mean = top_recent.groupby("item_id")["rating"].mean().to_dict()
    for it in items["item_id"]:
        speed_scores[it] = item_mean.get(it, 0.0)

def norm(v):
    if v.max() == v.min():
        return v
    return (v - v.min()) / (v.max() - v.min())

item_list = list(items["item_id"])
b_arr = np.array([batch_scores.get(it, 0) for it in item_list])
s_arr = np.array([speed_scores.get(it, 0) for it in item_list])
b_norm, s_norm = norm(b_arr), norm(s_arr)

final_scores = (1 - alpha)*b_norm + alpha*s_norm
ranked_idx = np.argsort(-final_scores)[:top_k]

results = []
for idx in ranked_idx:
    iid = item_list[idx]
    results.append({
        "item_id": iid,
        "title": items.loc[items["item_id"]==iid, "title"].values[0],
        "type": items.loc[items["item_id"]==iid, "type"].values[0],
        "dist_km": items.loc[items["item_id"]==iid, "city_dist_km"].values[0],
        "score": float(final_scores[idx])
    })

st.table(pd.DataFrame(results))
st.caption("El modelo combina aprendizaje batch histórico con valoraciones recientes en tiempo real.")
