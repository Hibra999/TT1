from neuralforecast.auto import AutoNHITS
from neuralforecast import NeuralForecast
from utilsforecast.plotting import plot_series
from neuralforecast.losses.pytorch import MAE
from ray import tune
def get_config(trial):
    return {
        "max_steps": 100,
        "input_size": trial.suggest_categorical("input_size", [24, 48, 96]),
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-4, 1e-2),
        "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
        "random_seed": 1,
    }
def forecast_model(train_df, h, num_sample, backend):
    model = AutoNHITS(
        h=h,
        loss=MAE(),
        config=get_config,  
        backend=backend,
        num_samples=num_sample
    )
    nf = NeuralForecast(models=[model], freq="H")
    nf.fit(df=train_df)
    Y_hat_df = nf.predict()
    return Y_hat_df