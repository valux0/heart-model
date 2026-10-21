import pandas as pd

# 1. Cargar el dataset
archivo = 'diamonds.csv'
datos = pd.read_csv(archivo)

# 2. Ver las columnas disponibles y las primeras 5 filas
print("--- Columnas en el dataset ---")
print(datos.columns)

print("\n--- Primeras filas ---")
print(datos.head())

print("\n--- Resumen estadístico ---")
print(datos.describe())
