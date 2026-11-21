import io
import os
from threading import Lock

import joblib
import numpy as np
import pandas as pd
from fastapi import File, HTTPException, UploadFile
from pydantic import Field, ValidationError


def register_retrain(
    app,
    input_schema,
    startup_pipeline,
    active_model_path,
):
    """Register CSV retraining on the existing FastAPI application."""
    retrain_lock = Lock()

    class TrainingRow(input_schema):
        weekly_sales_ksh: float = Field(ge=0, le=1_000_000)

    @app.post("/retrain")
    def retrain(file: UploadFile = File(...)):
        """Validate uploaded rows and continue training the SGD model."""
        try:
            if not file.filename or not file.filename.lower().endswith(".csv"):
                raise HTTPException(
                    status_code=400,
                    detail="Upload a CSV file.",
                )

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
                    # CSV cells are text; parse numbers before validation.
                    for field in [
                        "owner_age",
                        "hours_worked_last_week",
                        "weekly_sales_ksh",
                    ]:
                        record[field] = float(record[field])

                    employee_count = float(record["employees"])

                    if not employee_count.is_integer():
                        raise ValueError(
                            "employees must be a whole number"
                        )

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

            # Prevent simultaneous uploads from overwriting each other.
            with retrain_lock:
                candidate = joblib.load(active_model_path)

                # Preserve the fitted imputer, scaler, and encoder.
                transformed = candidate.named_steps[
                    "preprocessor"
                ].transform(new_inputs)

                regressor = candidate.named_steps["model"]

                # Continue training for one additional epoch.
                regressor.partial_fit(transformed, new_targets)

                if (
                    not np.isfinite(regressor.coef_).all()
                    or not np.isfinite(regressor.intercept_).all()
                    or not np.isfinite(
                        candidate.predict(new_inputs)
                    ).all()
                ):
                    raise HTTPException(
                        status_code=422,
                        detail=(
                            "Training produced invalid values; "
                            "model unchanged."
                        ),
                    )

                # Replace the active model only after writing succeeds.
                temporary_path = active_model_path.with_suffix(".tmp")

                try:
                    joblib.dump(candidate, temporary_path)
                    os.replace(temporary_path, active_model_path)
                finally:
                    temporary_path.unlink(missing_ok=True)

            return {
                "message": (
                    "Retraining complete. Predictions now use "
                    "the updated model."
                ),
                "rows_used": len(new_data),
                "model": type(regressor).__name__,
                "epochs_added": 1,
            }

        finally:
            file.file.close()