# -*- coding: utf-8 -*-
import pandas as pd
from pickle import load
from sklearn.pipeline import Pipeline


def make_inference(in_model: Pipeline, in_data: dict) -> dict[str, float]:
    X = pd.DataFrame([in_data])
    price = in_model.predict(X)[0]
    return {"price": round(float(price), 3)}


def load_model(path: str) -> Pipeline:
    with open(path, "rb") as file:
        model = load(file)
    return model
