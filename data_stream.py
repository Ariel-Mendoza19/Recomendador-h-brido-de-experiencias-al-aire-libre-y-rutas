import random
import time
import pandas as pd
import os

lugares = [
    "Parque Nacional Cajas",
    "Museo Pumapungo",
    "Mirador de Turi",
    "Río Tomebamba",
    "Parque Calderón",
    "Plaza de las Flores",
    "Laguna Llaviucu",
    "Museo de las Culturas Aborígenes"
]

def generate_stream():
    """Simula un flujo de calificaciones en tiempo real"""
    while True:
        user_id = f"user{random.randint(1, 5)}"
        lugar = random.choice(lugares)
        rating = random.randint(1, 5)
        yield {"user_id": user_id, "lugar": lugar, "rating": rating}
        time.sleep(2)  # simula llegada cada 2 segundos

def append_to_csv(data, file="stream_data.csv"):
    """Guarda los datos del flujo"""
    if not os.path.exists(file):
        df = pd.DataFrame(columns=["user_id", "lugar", "rating"])
    else:
        df = pd.read_csv(file)
    df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
    df.to_csv(file, index=False)
