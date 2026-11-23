
import pandas as pd
import numpy as np

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load prepared data
X = pd.read_csv("V0001_X.csv")
y_data = pd.read_csv("V0001_y.csv")

# Predict execution quality
y = y_data["execution_quality"]

# Leave-one-out cross-validation
cv = LeaveOneOut()

# Baseline: always predict the training-set average
baseline = DummyRegressor(strategy="mean")
baseline_predictions = cross_val_predict(baseline, X, y, cv=cv)

# AQA model: Ridge regression
model = make_pipeline(
    StandardScaler(),
    Ridge(alpha=1.0)
)

predictions = cross_val_predict(model, X, y, cv=cv)

# Evaluate both models
def evaluate(name, actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))

    print(name)
    print("MAE:", round(mae, 3))
    print("RMSE:", round(rmse, 3))
    print()

evaluate("Average-rating baseline", y, baseline_predictions)
evaluate("Ridge regression", y, predictions)

# Save predictions for inspection
results = pd.DataFrame({
    "actual_execution_quality": y,
    "baseline_prediction": baseline_predictions,
    "ridge_prediction": predictions
})

results.to_csv("V0001_baseline_predictions.csv", index=False)

print("Predictions saved to V0001_baseline_predictions.csv")
