# -*- coding: utf-8 -*-
import pytest
import pandas as pd
from sklearn.pipeline import Pipeline

from model_utils import make_inference, load_model


@pytest.fixture
def create_data():
    return {
        "car_ID": 1,
        "CarName": "mazda rx3",
        "symboling": 1,
        "fueltype": "gas",
        "aspiration": "std",
        "doornumber": "two",
        "carbody": "hatchback",
        "drivewheel": "fwd",
        "enginelocation": "front",
        "wheelbase": 93.1,
        "carlength": 159.1,
        "carwidth": 64.2,
        "carheight": 54.1,
        "curbweight": 1890,
        "enginetype": "ohc",
        "cylindernumber": "four",
        "enginesize": 91,
        "fuelsystem": "2bbl",
        "boreratio": 3.03,
        "stroke": 3.15,
        "compressionratio": 9.0,
        "horsepower": 68,
        "peakrpm": 5000,
        "citympg": 30,
        "highwaympg": 31
    }


def test_make_inference(monkeypatch, create_data):
    def mock_predict(self, X: pd.DataFrame):
        # просто проверяем что пришёл DataFrame
        assert isinstance(X, pd.DataFrame)
        return [12345.678]

    model = Pipeline([])
    monkeypatch.setattr(Pipeline, "predict", mock_predict)

    result = make_inference(model, create_data)
    assert result == {"price": 12345.678}


@pytest.fixture()
def filepath_and_data(tmpdir):
    import pickle

    p = tmpdir.mkdir("datadir").join("model.pkl")
    example = Pipeline([])
    p.write_binary(pickle.dumps(example))
    return str(p), example


def test_load_model(filepath_and_data):
    loaded = load_model(filepath_and_data[0])
    assert isinstance(loaded, Pipeline)