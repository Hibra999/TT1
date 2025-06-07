import pandas as pd
from utilsforecast.losses import mae, mse, rmse
from utilsforecast.evaluation import evaluate
from utils.helpers import API

nixtla_client = API()

def evalModelGPT(dataframe, porcentaje, freq, model):
    umbral_corte = int(len(dataframe) * porcentaje / 100)
    train = dataframe[:umbral_corte] 
    test = dataframe[umbral_corte:] 
    h = len(test)
    test = test.copy()
    depths = [1,2,3,4,5]
    for dep in depths:
        preds_df = nixtla_client.forecast(
            df=train,
            h=h,
            freq=freq,
            time_col="ds",
            target_col="y",
            model=model,
            level=[90],
            finetune_steps=5,
            finetune_loss="mae",
            finetune_depth=dep,
            add_history=True 
        )
        preds = preds_df["TimeGPT"].values
        test.loc[:,f"Depth = {dep} "] = preds
    
    test["unique_id"] = 0
    evaluationF = evaluate(test, metrics=[mae, mse, rmse], time_col="ds", target_col="y")
    return evaluationF, test