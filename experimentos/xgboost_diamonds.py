import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

# 1. Cargar datos
archivo = 'diamonds.csv'
datos = pd.read_csv(archivo)

# 2. Seleccionar variable objetivo (y) y características (X)
y = datos.price
# Para XGBoost clásico usamos solo las variables numéricas, o tendríamos que codificarlas
# Aquí tomamos el enfoque de solo usar las numéricas para simplificar (como en W5 de XGBoost)
X = datos.select_dtypes(include=['int64', 'float64']).drop(['price', 'Unnamed: 0'], axis=1)

# 3. Dividir los datos
train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state=0)

# 4. XGBoost Básico
print("--- XGBoost (W5) ---")
modelo_basico = XGBRegressor()
modelo_basico.fit(train_X, train_y)

predicciones_basicas = modelo_basico.predict(val_X)
mae_basico = mean_absolute_error(val_y, predicciones_basicas)
print(f"MAE (Modelo Básico): ${mae_basico:.2f} USD")

# 5. XGBoost Optimizado (Tuning de parámetros)
# n_estimators: Número de ciclos
# learning_rate: Tasa de aprendizaje
# early_stopping_rounds: Detener si no mejora (en XGBoost >= 3 se pasa al constructor)
modelo_optimizado = XGBRegressor(n_estimators=1000, 
                                 learning_rate=0.05, 
                                 n_jobs=4,
                                 early_stopping_rounds=5)

modelo_optimizado.fit(train_X, train_y,
                      eval_set=[(val_X, val_y)],
                      verbose=False)

predicciones_optimizadas = modelo_optimizado.predict(val_X)
mae_optimizado = mean_absolute_error(val_y, predicciones_optimizadas)
print(f"MAE (Modelo Optimizado): ${mae_optimizado:.2f} USD")

# 6. Prueba visual con el modelo optimizado
print("\n--- Predicciones de muestra (primeros 5 diamantes) ---")
comparacion = pd.DataFrame({'Real': val_y.head().values, 'Predicho': predicciones_optimizadas[:5].astype(int)})
comparacion['Error'] = abs(comparacion['Real'] - comparacion['Predicho'])
print(comparacion.to_string(index=False))
