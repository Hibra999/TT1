from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA

def forecast_model(train_df, h, freq, level=[90], **kwargs):

    arima_params = {
        'seasonal': False,  
        'stepwise': True,   
        'approximation': True,  
        'max_p': 5,       
        'max_q': 5,        
        'max_d': 2,        
        'start_p': 1,     
        'start_q': 1,          
        **kwargs
    }
    models = [AutoARIMA(**arima_params)]
    sf = StatsForecast(
        models=models,
        freq=freq,
        n_jobs=-1, 
    )
    forecast = sf.forecast(
        df=train_df,
        h=h,
        level=level
    )
    return forecast