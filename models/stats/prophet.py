from prophet import Prophet as ProphetDirect
import pandas as pd

def forecast_model(train_df, h, freq, level=[90], **kwargs):
    prophet_df = train_df[['ds', 'y']].copy()
    prophet_params = {
        'seasonality_mode': 'multiplicative',
        'changepoint_prior_scale': 0.1,
        'seasonality_prior_scale': 0.01,
        'holidays_prior_scale': 0.01,
        'n_changepoints': 25,
        'yearly_seasonality': False,
        'weekly_seasonality': True,
        'daily_seasonality': False,
        'interval_width': 0.95,
        **kwargs
    }
    model = ProphetDirect(**prophet_params)
    model.fit(prophet_df)
    future = model.make_future_dataframe(periods=h, freq=freq)
    forecast = model.predict(future)
    return model, forecast
    