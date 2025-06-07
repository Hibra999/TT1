import pandas as pd
from utilsforecast.losses import mae, mse, rmse
from utilsforecast.evaluation import evaluate
from utils.helpers import API

nixtla_client = API()

def timeGPTrico(dataframe, horizon, freq, model):
    h = horizon
    train_df = dataframe.iloc[:-h].copy()
    test  = dataframe.iloc[-h:].copy()
    forecast = nixtla_client.forecast(
        df=train_df,
        h=h,
        freq=freq,
        time_col="ds",
        target_col="y",
        model=model,
        level=[90],
        finetune_steps=200,
        finetune_loss="mae",
        finetune_depth=5,  
    )
    results = {
        'fcst': forecast,
        'train_data': test,
    }
    return results

def maeMSEetc(test, fcst):
    test = test.copy()
    test.loc[:, "TimeGPT"] = fcst["TimeGPT"].values
    return evaluate(test,metrics=[mae, mse, rmse], time_col="ds", target_col="y")