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



#PLOTS PARA EL BACKTESTNG(por el momento solo para TimeGPT)

def forecast_plot_evaluation(train_df, test_df_with_predictions, models_to_plot):
    combined_df = pd.concat([train_df, test_df_with_predictions], ignore_index=True)
    fcst_df = test_df_with_predictions[["ds", "unique_id"] + models_to_plot].copy()
    
    fig = nixtla_client.plot(
        combined_df,
        fcst_df,
        time_col="ds",
        target_col="y",
        level=[90],
        models=models_to_plot
    )
    return fig

def forecast_only_test_period(train_df, test_df_with_predictions, models_to_plot, umbral=30):
    train_context = train_df.tail(umbral)
    combined_df = pd.concat([train_context, test_df_with_predictions], ignore_index=True)
    fcst_df = test_df_with_predictions[["ds", "unique_id"] + models_to_plot].copy()
    
    fig = nixtla_client.plot(
        combined_df,
        fcst_df,
        time_col="ds",
        target_col="y",
        level=[90],
        models=models_to_plot
    )
    return fig