# -*- coding: utf-8 -*-
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def init_test_client(monkeypatch) -> TestClient:
    def mock_make_inference(*args, **kwargs) -> dict[str, float]:
        return {"price": 13495.0}

    def mock_load_model(*args, **kwargs):
        return None

    monkeypatch.setenv("MODEL_PATH", "faked/model.pkl")
    monkeypatch.setattr("model_utils.make_inference", mock_make_inference)
    monkeypatch.setattr("model_utils.load_model", mock_load_model)

    from main import app
    return TestClient(app)


@pytest.fixture
def valid_payload() -> dict:
    return {
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


def test_healthcheck(init_test_client) -> None:
    response = init_test_client.get("/healthcheck")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_token_correctness(init_test_client, valid_payload) -> None:
    response = init_test_client.post(
        "/predictions",
        headers={"Authorization": "Bearer 00000"},
        json=valid_payload
    )
    assert response.status_code == 200
    assert "price" in response.json()


def test_token_not_correctness(init_test_client, valid_payload) -> None:
    response = init_test_client.post(
        "/predictions",
        headers={"Authorization": "Bearer kedjkj"},
        json=valid_payload
    )
    assert response.status_code == 401
    assert response.json() == {
        "detail": "Invalid authentication credentials"
    }


def test_token_absent(init_test_client, valid_payload) -> None:
    response = init_test_client.post(
        "/predictions",
        json=valid_payload
    )
    assert response.status_code == 401
    assert response.json() == {
        "detail": "Not authenticated"
    }


def test_inference(init_test_client, valid_payload) -> None:
    response = init_test_client.post(
        "/predictions",
        headers={"Authorization": "Bearer 00000"},
        json=valid_payload
    )
    assert response.status_code == 200
    assert response.json()["price"] == 13495.0
