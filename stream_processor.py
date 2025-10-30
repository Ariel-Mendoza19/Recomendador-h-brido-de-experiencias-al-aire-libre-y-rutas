import pandas as pd
import os

def update_recommendations(user_id, file="stream_data.csv"):
    """Procesa el flujo y genera recomendaciones en tiempo real"""
    if not os.path.exists(file):
        return ["No hay datos disponibles aún"]

    df = pd.read_csv(file)

    # Calcular promedios dinámicamente (solo con los datos del stream)
    promedios = df.groupby("lugar")["rating"].mean().reset_index()
    promedios = promedios.sort_values(by="rating", ascending=False)

    # Filtrar lugares ya calificados por el usuario
    if user_id in df["user_id"].values:
        calificados = df[df["user_id"] == user_id]["lugar"].unique()
        sugerencias = promedios[~promedios["lugar"].isin(calificados)]
    else:
        sugerencias = promedios

    return sugerencias.head(5)["lugar"].tolist()
