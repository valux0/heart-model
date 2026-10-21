import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# 1. Cargar datos
archivo = 'diamonds.csv'
datos = pd.read_csv(archivo)

# 2. Limpieza básica de datos faltantes (filas vacías)
datos = datos.dropna(axis=0)

# 3. Seleccionar la variable objetivo (y)
y = datos.price

# 4. Seleccionar las características numéricas (X)
caracteristicas = ['carat', 'depth', 'table', 'x', 'y', 'z']
X = datos[caracteristicas]

# 5. Dividir los datos en entrenamiento (train) y validación (val)
# Se usa un 80% para entrenar y 20% para validar
train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state=0)

# 6. Definir el modelo (Árbol de Decisión para Regresión)
modelo_diamantes = DecisionTreeRegressor(random_state=1)

# 7. Entrenar el modelo con los datos de entrenamiento
print("Entrenando el modelo...")
modelo_diamantes.fit(train_X, train_y)
print("¡Entrenamiento finalizado!")

# 8. Hacer una pequeña prueba visual de predicción con 5 ejemplos
print("\n--- Predicciones de muestra (primeros 5 diamantes de validación) ---")
print("Características:")
print(val_X.head())

predicciones_muestra = modelo_diamantes.predict(val_X.head())
print("\nPrecios predichos:")
print(predicciones_muestra)

print("\nPrecios reales:")
print(val_y.head().values)

# 9. Evaluar el modelo con la métrica MAE (Mean Absolute Error)
predicciones_completas = modelo_diamantes.predict(val_X)
mae = mean_absolute_error(val_y, predicciones_completas)

print("\n--- Evaluación global ---")
print(f"Error absoluto medio (MAE) en validación: ${mae:.2f} USD")
