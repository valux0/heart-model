from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from joblib import load
import pathlib
from fastapi.middleware.cors import CORSMiddleware

# 1. Configuración de la aplicación FastAPI
app = FastAPI(title='Diamond Price Prediction')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# 2. Cargar el modelo entrenado
model = load(pathlib.Path('model/diamond-model-v1.joblib'))

# 3. Definir el esquema de datos que recibirá la API
class InputData(BaseModel):
    carat: float = 1.10
    cut: str = "Ideal"
    color: str = "H"
    clarity: str = "SI2"
    depth: float = 62.0
    table: float = 55.0
    x: float = 6.61
    y: float = 6.65
    z: float = 4.11

# 4. Definir el formato de salida
class OutputData(BaseModel):
    score: float = 0.0

# 5. Endpoint POST /score
@app.post('/score', response_model=OutputData)
def score(data: InputData):
    # Convertimos los datos de entrada a DataFrame para que el Pipeline los procese
    input_df = pd.DataFrame([data.dict()])
    result = model.predict(input_df)[0]
    return {'score': float(result)}
