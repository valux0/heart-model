import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from joblib import dump
import pathlib

# 1. Cargar datos desde data/
df = pd.read_csv(pathlib.Path('data/diamonds.csv'))
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

# 2. Separar target (y) y features (X)
y = df.pop('price')
X = df

# 3. Columnas numéricas y categóricas
categorical_cols = ['cut', 'color', 'clarity']
numerical_cols = ['carat', 'depth', 'table', 'x', 'y', 'z']

# Preprocesamiento
numerical_transformer = SimpleImputer(strategy='median')
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numerical_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])

# 4. Pipeline completo
# Cambiar n_estimators=100 por un modelo liviano para no saturar la RAM de Render
pipeline_model = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        (
            'model',
            RandomForestRegressor(
                n_estimators=15, max_depth=12, random_state=0
            ),
        ),
    ]
)

# 5. Entrenar modelo
print("Entrenando modelo...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
pipeline_model.fit(X_train, y_train)

# 6. Guardar en model/diamond-model-v1.joblib
output_path = pathlib.Path('model/diamond-model-v1.joblib')
print(f"Guardando modelo en {output_path}...")
dump(pipeline_model, output_path)
print("¡Modelo guardado exitosamente!")