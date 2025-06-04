from neuralforecast.auto import AutoNHITS
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import MAE

def forecast_model(train_df, h, num_sample, backend):
    forecast = AutoNHITS(
        h=h,
        loss=MAE(),
        config=None,  #Con optuna, automaticamente busca los mejores hiperparametros
        backend=backend,
        num_samples=num_sample,
        gpus=0,
        cpus=16
    )
    nf = NeuralForecast(models=[forecast], freq="h")
    nf.fit(df=train_df)
    y_hat = nf.predict()
    return y_hat