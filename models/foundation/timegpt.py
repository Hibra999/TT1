from utils.helpers import API
import matplotlib.pyplot as plt
nixtla_client = API()

def forecast_model(train_df, h, freq, model="timegpt-1", **kwargs):

    forecast = nixtla_client.forecast(
        df=train_df,
        h=h,
        freq=freq,
        time_col="ds",
        target_col="y",
        model=model,
        level=[90],
        finetune_steps=500,
        finetune_loss="mae",
        finetune_depth=5,
        add_history=True,
        **kwargs    
    )
    return forecast

