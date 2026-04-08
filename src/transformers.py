# -*- coding: utf-8 -*-
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        X["brand"] = X["CarName"].apply(lambda x: x.split(" ")[0])
        X["model"] = X["CarName"].apply(lambda x: " ".join(x.split(" ")[1:]))

        numerical_columns = [
            "wheelbase", "carlength", "carwidth", "carheight", "curbweight",
            "enginesize", "boreratio", "stroke", "compressionratio",
            "horsepower", "peakrpm", "citympg", "highwaympg"
        ]

        X["power_to_weight_ratio"] = X["horsepower"] / X["curbweight"]

        for column in numerical_columns:
            X[f"{column}_squared"] = X[column] ** 2

        X["log_enginesize"] = np.log(X["enginesize"] + 1)

        return X