from neuralforecast.auto import AutoNHITS
from neuralforecast import NeuralForecast
from utilsforecast.plotting import plot_series
from ray import tune
def forecast_model(train_df, h, num_sample, backend):
    model = AutoNHITS(
        h=h,
        num_samples=num_sample,
        backend=backend,
        config=dict(
            max_steps=100,
            input_size=tune.choice([3*len(train_df)]),
            learning_rate=tune.choice([1e-3])
        )
    )
    nf = NeuralForecast(models=[model], freq=h)
    nf.fit(df=train_df)
    Y_hat_df = nf.predict() 
    return Y_hat_df