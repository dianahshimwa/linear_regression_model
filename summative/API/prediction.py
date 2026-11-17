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

# Allow only the listed local browser origins during development.
# Restrict methods and headers to those needed by the API; credentials
# are disabled because the API does not use browser cookies.
# Native Flutter mobile requests are not restricted by browser CORS.
# Add the actual Render origin to this list when deploying.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5000",
        "http://127.0.0.1:5000",
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

class TrainingRow(PredictionInput):
    weekly_sales_ksh: float = Field(ge=0, le=1_000_000)


@app.post("/retrain")
def retrain(file: UploadFile = File(...)):
    """Continue SGD training using validated, cleaned CSV rows."""
    try:
        if not file.filename or not file.filename.lower().endswith(".csv"):
            raise HTTPException(
                status_code=400,
                detail="Upload a CSV file.",
            )

        # Limit uploads to 2 MB.
        content = file.file.read(2_000_001)
        if len(content) > 2_000_000:
            raise HTTPException(
                status_code=413,
                detail="CSV must be no larger than 2 MB.",
            )

        try:
            frame = pd.read_csv(
                io.BytesIO(content),
                dtype=str,
                keep_default_na=False,
                encoding="utf-8-sig",
            )
        except (ValueError, UnicodeError, pd.errors.ParserError):
            raise HTTPException(
                status_code=400,
                detail="The file could not be read as a UTF-8 CSV.",
            )

        feature_names = list(startup_pipeline.feature_names_in_)
        expected_columns = feature_names + ["weekly_sales_ksh"]

        if set(frame.columns) != set(expected_columns):
            raise HTTPException(
                status_code=422,
                detail={
                    "message": "CSV columns must match the training schema.",
                    "required_columns": expected_columns,
                },
            )

        if frame.empty or len(frame) > 10_000:
            raise HTTPException(
                status_code=422,
                detail="Provide between 1 and 10,000 data rows.",
            )

        validated_rows = []

        for row_number, record in enumerate(
            frame.to_dict(orient="records"), start=2
        ):
            try:
                # CSV has no numeric types, so parse numeric cells
                # explicitly before applying the strict request schema.
                for field in [
                    "owner_age",
                    "hours_worked_last_week",
                    "weekly_sales_ksh",
                ]:
                    record[field] = float(record[field])

                employee_count = float(record["employees"])
                if not employee_count.is_integer():
                    raise ValueError("employees must be a whole number")
                record["employees"] = int(employee_count)

                validated = TrainingRow.model_validate(record)
                validated_rows.append(validated.model_dump())

            except (ValueError, OverflowError, ValidationError) as exc:
                raise HTTPException(
                    status_code=422,
                    detail=f"Invalid CSV row {row_number}: {exc}",
                )

        new_data = pd.DataFrame(validated_rows)
        new_inputs = new_data[feature_names]
        new_targets = new_data["weekly_sales_ksh"]

        # Serialize updates so simultaneous uploads cannot overwrite
        # each other's training changes.
        with retrain_lock:
            candidate = joblib.load(ACTIVE_MODEL_PATH)

            # Keep imputation, scaling, and encoding fixed.
            transformed = candidate.named_steps[
                "preprocessor"
            ].transform(new_inputs)

            regressor = candidate.named_steps["model"]

            # One additional epoch on new observations.
            regressor.partial_fit(transformed, new_targets)

            if (
                not np.isfinite(regressor.coef_).all()
                or not np.isfinite(regressor.intercept_).all()
                or not np.isfinite(candidate.predict(new_inputs)).all()
            ):
                raise HTTPException(
                    status_code=422,
                    detail="Training produced invalid values; model unchanged.",
                )

            # Write fully before replacing the active model.
            temporary_path = ACTIVE_MODEL_PATH.with_suffix(".tmp")
            try:
                joblib.dump(candidate, temporary_path)
                os.replace(temporary_path, ACTIVE_MODEL_PATH)
            finally:
                temporary_path.unlink(missing_ok=True)

        return {
            "message": "Retraining complete. Predictions now use the updated model.",
            "rows_used": len(new_data),
            "model": type(regressor).__name__,
            "epochs_added": 1,
        }

    finally:
        file.file.close()