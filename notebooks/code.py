import numpy as np
import pandas as pd
import pickle

from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.transformers import FeatureEngineer




# Загрузка данных
df = pd.read_csv("../data/CarPrice_Assignment.csv")

X = df.drop(columns=["price", "car_ID"])
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

categorical_columns = [
    "fueltype", "aspiration", "doornumber", "carbody", "drivewheel",
    "enginelocation", "enginetype", "cylindernumber", "fuelsystem",
    "brand", "model"
]

numerical_columns = [
    "symboling", "wheelbase", "carlength", "carwidth", "carheight",
    "curbweight", "enginesize", "boreratio", "stroke", "compressionratio",
    "horsepower", "peakrpm", "citympg", "highwaympg",
    "power_to_weight_ratio", "log_enginesize",
    "wheelbase_squared", "carlength_squared", "carwidth_squared",
    "carheight_squared", "curbweight_squared", "enginesize_squared",
    "boreratio_squared", "stroke_squared", "compressionratio_squared",
    "horsepower_squared", "peakrpm_squared", "citympg_squared",
    "highwaympg_squared"
]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_columns),
        (
            "cat",
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            categorical_columns
        ),
    ],
    remainder="drop"
)

pipeline = Pipeline([
    ("feature_engineering", FeatureEngineer()),
    ("preprocessor", preprocessor),
    ("model", LinearRegression())
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R-squared: {r2:.2f}")

Path("../models").mkdir(parents=True, exist_ok=True)

with open("../models/pipeline.pkl", "wb") as f:
    pickle.dump(pipeline, f)