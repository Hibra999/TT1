from utils.helpers import API
nixtla_client = API()

def normal_plot(df):
    fig = nixtla_client.plot(df, time_col='ds', target_col='y')  
    return fig

def forecast_plot(final_df, fcst_df, model):

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
