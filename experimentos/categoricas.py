import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

# 1. Cargar datos
archivo = 'diamonds.csv'
datos = pd.read_csv(archivo)

# 2. Seleccionar variable objetivo (y) y características (X)
y = datos.price
X = datos.drop(['price', 'Unnamed: 0'], axis=1) # Excluimos 'Unnamed: 0' que es un índice

# 3. Dividir los datos
train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state=0)

# Detectar variables categóricas (las que son de tipo object/string)
columnas_categoricas = [col for col in train_X.columns if train_X[col].dtype.kind == 'O']
columnas_numericas = [col for col in train_X.columns if train_X[col].dtype.kind in ('i', 'f')]

print("--- Variables Categóricas (W5) ---")
print(f"Detectadas: {columnas_categoricas}\n")

# Función para evaluar
def evaluar_modelo(train_X_param, val_X_param, train_y_param, val_y_param):
    modelo = RandomForestRegressor(n_estimators=10, random_state=0)
    modelo.fit(train_X_param, train_y_param)
    predicciones = modelo.predict(val_X_param)
    return mean_absolute_error(val_y_param, predicciones)

# Enfoque 1: Eliminar variables categóricas
train_X_drop = train_X.select_dtypes(exclude=['object', 'string'])
val_X_drop = val_X.select_dtypes(exclude=['object', 'string'])
mae_drop = evaluar_modelo(train_X_drop, val_X_drop, train_y, val_y)
print(f"MAE Enfoque 1 (Eliminar Categóricas): ${mae_drop:.2f}")

# Enfoque 2: Codificación Ordinal (Ordinal Encoding)
train_X_ordinal = train_X.copy()
val_X_ordinal = val_X.copy()
codificador_ordinal = OrdinalEncoder()
train_X_ordinal[columnas_categoricas] = codificador_ordinal.fit_transform(train_X[columnas_categoricas])
val_X_ordinal[columnas_categoricas] = codificador_ordinal.transform(val_X[columnas_categoricas])
mae_ordinal = evaluar_modelo(train_X_ordinal, val_X_ordinal, train_y, val_y)
print(f"MAE Enfoque 2 (Codificación Ordinal): ${mae_ordinal:.2f}")

# Enfoque 3: Codificación One-Hot (One-Hot Encoding)
codificador_oh = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
train_X_oh_cols = pd.DataFrame(codificador_oh.fit_transform(train_X[columnas_categoricas]))
val_X_oh_cols = pd.DataFrame(codificador_oh.transform(val_X[columnas_categoricas]))

train_X_oh_cols.index = train_X.index
val_X_oh_cols.index = val_X.index

train_X_num = train_X.drop(columnas_categoricas, axis=1)
val_X_num = val_X.drop(columnas_categoricas, axis=1)

train_X_oh = pd.concat([train_X_num, train_X_oh_cols], axis=1)
val_X_oh = pd.concat([val_X_num, val_X_oh_cols], axis=1)

train_X_oh.columns = train_X_oh.columns.astype(str)
val_X_oh.columns = val_X_oh.columns.astype(str)
mae_oh = evaluar_modelo(train_X_oh, val_X_oh, train_y, val_y)
print(f"MAE Enfoque 3 (One-Hot Encoding): ${mae_oh:.2f}")

# 4. Mostrar una prueba visual (con One-Hot que suele ser el mejor)
modelo_oh = RandomForestRegressor(n_estimators=10, random_state=0)
modelo_oh.fit(train_X_oh, train_y)
predicciones_oh = modelo_oh.predict(val_X_oh.head())

print("\n--- Predicciones de muestra (primeros 5 diamantes usando One-Hot) ---")
comparacion = pd.DataFrame({'Real': val_y.head().values, 'Predicho': predicciones_oh.astype(int)})
comparacion['Error'] = abs(comparacion['Real'] - comparacion['Predicho'])
print(comparacion.to_string(index=False))
