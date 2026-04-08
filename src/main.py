# -*- coding: utf-8 -*-
import os
from src.model_utils import load_model, make_inference
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel


class Instance(BaseModel):
    CarName: str
    symboling: int
    fueltype: str
    aspiration: str
    doornumber: str
    carbody: str
    drivewheel: str
    enginelocation: str
    wheelbase: float
    carlength: float
    carwidth: float
    carheight: float
    curbweight: int
    enginetype: str
    cylindernumber: str
    enginesize: int
    fuelsystem: str
    boreratio: float
    stroke: float
    compressionratio: float
    horsepower: int
    peakrpm: int
    citympg: int
    highwaympg: int


app = FastAPI()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth")
model_path: str = os.getenv("MODEL_PATH")
if model_path is None:
    raise ValueError("The environment variable $MODEL_PATH is empty!")


async def is_token_correct(token: str) -> bool:
    return token == "00000"


async def check_token(token: str = Depends(oauth2_scheme)) -> None:
    if not await is_token_correct(token):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


@app.get("/healthcheck")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predictions")
async def predictions(
    instance: Instance,
    token: str = Depends(check_token)
) -> dict[str, float]:
    model = load_model(model_path)
    return make_inference(model, instance.model_dump())
