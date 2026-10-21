import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer

# 1. Cargar datos
archivo = 'diamonds.csv'
datos = pd.read_csv(archivo)

# 2. Seleccionar variable objetivo y características numéricas
y = datos.price
caracteristicas = ['carat', 'depth', 'table', 'x', 'y', 'z']
X = datos[caracteristicas]

# 3. Dividir los datos
train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state=0)

# Función para evaluar con Random Forest
def evaluar_modelo(train_X, val_X, train_y, val_y):
    modelo = RandomForestRegressor(n_estimators=10, random_state=0)
    modelo.fit(train_X, train_y)
    predicciones = modelo.predict(val_X)
    return mean_absolute_error(val_y, predicciones)

print("--- Random Forest y Valores Nulos ---")
print("Nota: El dataset de diamantes no tiene valores nulos,")
print("pero aplicamos las 3 estrategias de la clase (W5) para mantener la práctica.\n")

columnas_con_nulos = [col for col in train_X.columns if train_X[col].isnull().any()]

# Enfoque 1: Eliminar columnas con valores faltantes
train_X_reducido = train_X.drop(columnas_con_nulos, axis=1)
val_X_reducido = val_X.drop(columnas_con_nulos, axis=1)
mae_1 = evaluar_modelo(train_X_reducido, val_X_reducido, train_y, val_y)
print(f"MAE Enfoque 1 (Eliminar columnas): ${mae_1:.2f}")

# Enfoque 2: Imputación (Rellenar valores)
imputador = SimpleImputer()
train_X_imputado = pd.DataFrame(imputador.fit_transform(train_X))
val_X_imputado = pd.DataFrame(imputador.transform(val_X))
train_X_imputado.columns = train_X.columns
val_X_imputado.columns = val_X.columns
mae_2 = evaluar_modelo(train_X_imputado, val_X_imputado, train_y, val_y)
print(f"MAE Enfoque 2 (Imputación Simple): ${mae_2:.2f}")

# Enfoque 3: Extensión a Imputación
train_X_plus = train_X.copy()
val_X_plus = val_X.copy()
for col in columnas_con_nulos:
    train_X_plus[col + '_faltaba'] = train_X_plus[col].isnull()
    val_X_plus[col + '_faltaba'] = val_X_plus[col].isnull()

train_X_imputado_plus = pd.DataFrame(imputador.fit_transform(train_X_plus))
val_X_imputado_plus = pd.DataFrame(imputador.transform(val_X_plus))
train_X_imputado_plus.columns = train_X_plus.columns
val_X_imputado_plus.columns = val_X_plus.columns
mae_3 = evaluar_modelo(train_X_imputado_plus, val_X_imputado_plus, train_y, val_y)
print(f"MAE Enfoque 3 (Imputación Extendida): ${mae_3:.2f}")

# 4. Mostrar una prueba visual
modelo_rf = RandomForestRegressor(n_estimators=10, random_state=0)
modelo_rf.fit(train_X_imputado, train_y)
predicciones_rf = modelo_rf.predict(val_X_imputado.head())

print("\n--- Predicciones de muestra (primeros 5 diamantes) ---")
comparacion = pd.DataFrame({'Real': val_y.head().values, 'Predicho': predicciones_rf.astype(int)})
comparacion['Error'] = abs(comparacion['Real'] - comparacion['Predicho'])
print(comparacion.to_string(index=False))
