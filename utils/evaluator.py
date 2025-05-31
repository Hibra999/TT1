from utils import helpers  
import pandas as pd

def evaluate_forecast(test_df: pd.DataFrame, forecast_df: pd.DataFrame, model_name: str) -> dict:
    merged = test_df.merge(forecast_df, on="ds")
    results = {
        "mae": helpers.mae(merged['y'], merged[model_name]),
        "rmse": helpers.rmse(merged['y'], merged[model_name]),
        "smape": helpers.smape(merged['y'], merged[model_name])
    }
    return results