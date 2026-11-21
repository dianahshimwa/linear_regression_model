## 1. Mission and problem

My mission is to empower women in Africa, starting in Rwanda, through technology.
This project estimates weekly business sales to support women entrepreneurs with planning.
It uses historical Kenyan data and requires local validation before use in Rwanda.

## 2. Dataset and source

The dataset comes from David McKenzie and Susana Puerto’s study,
“Growing Markets through Business Training for Female Entrepreneurs:
A Market-Level Randomized Experiment in Kenya.”

The baseline dataset contains 3,537 observations and 959 columns.
After cleaning, 3,065 observations were retained. The model uses eight
numeric and categorical inputs to estimate reported weekly sales in
Kenyan shillings.

Dataset: https://doi.org/10.3886/E119063V1

Four regression models were compared using the same test set:
SGDRegressor, LinearRegression, DecisionTreeRegressor, and RandomForestRegressor.

SGDRegressor had the lowest test MSE: 131,348,756.35 KSh².
Its test RMSE was 11,460.75 KSh. The saved pipeline includes preprocessing.

This is an educational sales-estimation prototype, not a future-sales
forecast. Estimates are not guaranteed business outcomes.

## 3. Public Swagger UI

https://women-enterprise-sales.onrender.com/docs

Available routes:

- `POST /predict`: validate eight inputs and estimate weekly sales.
- `POST /retrain`: upload a CSV to continue training the SGD model.
- `GET /docs`: interactive API documentation.

The free Render service may take around a minute to wake after inactivity.

Retraining CSVs must contain these columns:

```text
owner_age,employees,hours_worked_last_week,education_level,county,business_type,advertises,keeps_written_records,weekly_sales_ksh
```

Use cleaned values matching the prediction schema, with the observed sales
target added. Each accepted upload runs one additional epoch using
`partial_fit`, while keeping the fitted preprocessing unchanged.
Predictions immediately use the updated runtime model. Restarting the
service restores the original repository model.

Render configuration:

- Root directory: `summative/API`
- Python version: `3.13.18`
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn prediction:app --host 0.0.0.0 --port $PORT --workers 1`

Python dependencies are managed with uv. The repository includes
`pyproject.toml`, `uv.lock`, and exported API requirements.

## 4. YouTube video demo

https://youtu.be/UeDlqzQnVY8

## 5. Run the mobile app

### Prerequisites

- Git and Flutter installed.
- Android Studio with an Android emulator, or macOS with Xcode and an
  iOS simulator.
- An internet connection to access the deployed API.

### Instructions

1. Clone the repository:

   ```bash
   git clone https://github.com/dianahshimwa/linear_regression_model.git
   cd linear_regression_model/summative/FlutterApp
   ```

2. Install Flutter dependencies:

   ```bash
   flutter pub get
   ```

3. Start an Android emulator or iOS simulator.

4. List available devices:

   ```bash
   flutter devices
   ```

5. Run the app using the mobile device ID shown:

   ```bash
   flutter run -d <device-id>
   ```

6. Use the prefilled example or enter all eight business inputs, then tap
   **Predict**. The result appears below the button in Kenyan shillings.

7. To test validation, clear an input, enter an invalid number, or enter
   working hours above 168. A readable error appears in the same result area.

The API URL is configured in `lib/api_service.dart`.