from utils.helpers import API
from utilsforecast.plotting import plot_series
import matplotlib.pyplot as plt
import pandas as pd
nixtla_client = API()

def normal_plot(df):
    fig = nixtla_client.plot(df, time_col='ds', target_col='y')  
    return fig

def forecast_plot(final_df, fcst_df, model): # funcionan para las foundation

    fig = nixtla_client.plot(
        final_df,
        fcst_df,
        time_col="ds",
        target_col="y",
        level=[90],
        models=model
    )
    return fig

def forecast_only_future(final_df,fcst_df, h, model, umbral=30):

    fig = nixtla_client.plot(
        final_df.tail(umbral),
        fcst_df.tail(umbral + h),
        models=model,
        level=[90],
        time_col="ds",
        target_col="y",
    )
    return fig 

def plot_dl(train_df, futuro_df): # estas son para las autos
    fig = plot_series(train_df, futuro_df)
    return fig

def plot_dl_future_only(train_df, futuro_df, h, umbral=30): # estas son para las autos
    train_subset = train_df.tail(umbral)
    futuro_subset = futuro_df.tail(umbral + h)
    fig = plot_series(train_subset, futuro_subset)
    return fig

def plot_prophet_forecast(model, forecast): #Solo para prophet
    fig = model.plot(forecast)
    return fig

def plot_prophet_future_only(train_df, forecast_df, h, umbral=30): #Igual solamente para prophet
    train_subset = train_df.tail(umbral)
    future_subset = forecast_df.tail(h)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(train_subset['ds'], train_subset['y'], 
           label='Histórico', color='black', linewidth=2)
    ax.plot(future_subset['ds'], future_subset['yhat'], 
           label='Predicción', color='red', linewidth=2)
    ax.fill_between(future_subset['ds'], 
                   future_subset['yhat_lower'], 
                   future_subset['yhat_upper'],
                   alpha=0.3, color='blue', label='Intervalo 95%')
    ax.set_xlabel('ds')
    ax.set_ylabel('y')
    ax.set_title('Forecast Prophet')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


#PLOTS PARA EL BACKTESTNG(por el momento solo para TimeGPT)

def forecast_plot_evaluation(train_df, test_df_with_predictions, models_to_plot):
    filtered_test_df = test_df_with_predictions[['unique_id', 'ds', 'y'] + models_to_plot]
    return plot_series(train_df, filtered_test_df)

def forecast_only_test_period(train_df, test_df_with_predictions, models_to_plot, umbral):
    h = len(test_df_with_predictions)
    tail_train = train_df.tail(h).copy()
    tail_train['unique_id'] = test_df_with_predictions['unique_id'].iloc[0]
    forecasts_df = test_df_with_predictions[
        ['unique_id', 'ds', 'y'] + models_to_plot
    ].copy()
    fig = plot_series(
        df=tail_train,
        forecasts_df=forecasts_df,
        models=models_to_plot,
        plot_random=False,
        ids=[tail_train['unique_id'].iloc[0]]
    )
    return fig

def plot_backtestGPT(df_test, fcst):
    fig = nixtla_client.plot(df_test, fcst, time_col="ds", target_col="y")
    return fig