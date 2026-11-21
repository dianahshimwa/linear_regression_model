from pathlib import Path
from typing import Literal
import math

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field, field_validator

import io
import os
import shutil
import tempfile
from threading import Lock

import numpy as np
from fastapi import File, UploadFile
from pydantic import ValidationError

from retrain import register_retrain


MODEL_PATH = Path(__file__).resolve().parent / "best_model.pkl"

# Load at startup to detect a missing or incompatible model immediately.
startup_pipeline = joblib.load(MODEL_PATH)

# Preserve the repository model and update a temporary runtime copy.
runtime_directory = tempfile.TemporaryDirectory(
    prefix="women-enterprise-"
)
ACTIVE_MODEL_PATH = (
    Path(runtime_directory.name) / "active_model.pkl"
)
shutil.copyfile(MODEL_PATH, ACTIVE_MODEL_PATH)

retrain_lock = Lock()

encoder = (
    startup_pipeline.named_steps["preprocessor"]
    .named_transformers_["categorical"]
    .named_steps["encoder"]
)

# Matches categorical_features in the notebook:
# education, county, business type, advertising, record keeping.
KNOWN_BUSINESS_TYPES = {
    str(value) for value in encoder.categories_[2]
}

app = FastAPI(
    title="Women Enterprise Sales API",
    description=(
        "Estimate weekly sales in Kenyan shillings using historical "
        "Kenyan business data. Educational prototype."
    ),
    version="1.0.0",
)

# Allow only the listed local development origins and deployed Render origin.
# Restrict methods to GET/POST and headers to Content-Type.
# Credentials are disabled because this API does not use browser cookies.
# Browser CORS does not restrict native Flutter mobile requests.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5000",
        "http://127.0.0.1:5000",
        "https://women-enterprise-sales.onrender.com",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
    allow_credentials=False,
)


class PredictionInput(BaseModel):
    model_config = ConfigDict(
        strict=True,
        extra="forbid",
        allow_inf_nan=False,
    )

    owner_age: float = Field(ge=18, le=100)
    employees: int = Field(ge=0, le=3)
    hours_worked_last_week: float = Field(ge=0, le=168)

    education_level: str = Field(
        pattern=r"^level_([0-9]|1[0-8])$"
    )
    county: Literal["KITUI", "KISII", "KAKAMEGA", "EMBU"]
    business_type: str = Field(min_length=1, max_length=200)
    advertises: Literal["yes", "no"]
    keeps_written_records: Literal["yes", "no"]

    @field_validator("business_type")
    @classmethod
    def check_business_type(cls, value):
        if value not in KNOWN_BUSINESS_TYPES:
            raise ValueError(
                "Business type must match a category in the training data."
            )
        return value


class PredictionOutput(BaseModel):
    predicted_weekly_sales_ksh: float
    currency: str = "KES"


# Reuses the notebook's load-and-predict function.
def predict_one_row(row, model_path=ACTIVE_MODEL_PATH):
    """Load the saved pipeline and predict sales for one input row."""
    pipeline = joblib.load(model_path)

    if isinstance(row, pd.Series):
        row = row.to_frame().T
    elif isinstance(row, dict):
        row = pd.DataFrame([row])

    if not isinstance(row, pd.DataFrame) or len(row) != 1:
        raise ValueError("Provide exactly one row.")

    required_features = list(pipeline.feature_names_in_)
    return float(pipeline.predict(row[required_features])[0])


@app.post("/predict", response_model=PredictionOutput)
def predict(payload: PredictionInput):
    prediction = predict_one_row(payload.model_dump())

    if not math.isfinite(prediction):
        raise HTTPException(
            status_code=500,
            detail="The model returned a non-finite prediction.",
        )

    return PredictionOutput(
        predicted_weekly_sales_ksh=round(prediction, 2)
    )

register_retrain(
    app=app,
    input_schema=PredictionInput,
    startup_pipeline=startup_pipeline,
    active_model_path=ACTIVE_MODEL_PATH,
)