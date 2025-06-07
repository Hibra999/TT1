import streamlit as st
import matplotlib.pyplot as plt
import os
import sys
import asyncio
if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
from utils.data import get_ohlcv_dataframe, get_final_dataframe
from utils.plots import normal_plot, forecast_plot, forecast_only_future, plot_dl, plot_dl_future_only, forecast_plot_evaluation, forecast_only_test_period
from models.foundation import timegpt, timegpt_long
from models.dl import auto_nhits, auto_tft
from utils.evaluator import evalModelGPT
st.set_page_config(layout="wide")

token = st.selectbox("Token", ["BTC/USDT", "EUR/USDT"])

@st.cache_data
def load_ohlcv_cached(symbol: str, timeframe: str, since: str):
    return get_ohlcv_dataframe("binance", symbol, timeframe, since)
@st.cache_data
def load_final_df_cached(raw_df):
    return get_final_dataframe(raw_df)
    
df = load_ohlcv_cached(token, "1h", "2010-01-01 00:00:00")
st.dataframe(df.head(3))
final_df = get_final_dataframe(df) 
st.write("Datos procesados")
st.dataframe(final_df.head(3))
st.pyplot(normal_plot(final_df))

MODELOS_DIR = "models" 
categorias = []
for entry in os.listdir(MODELOS_DIR):
    full_path = os.path.join(MODELOS_DIR, entry)
    if os.path.isdir(full_path):
        categorias.append(entry)

categoria_sel = st.selectbox("Categoria", categorias)
modelos_disponibles = []
cat_path = os.path.join(MODELOS_DIR, categoria_sel)
for fname in os.listdir(cat_path):
    if fname.endswith(".py") and not fname.startswith("__"):
        modelos_disponibles.append(fname[:-3])  

modelo_sel = st.selectbox("Selecciona un modelo", modelos_disponibles)
umbral = st.number_input("Umbral", 30)
st.write("Hora, Dia, Semana, Mes")
h =  st.selectbox("Horizonte, hrs", [1, 24, 168, 730])

if 'modelo_anterior' not in st.session_state: #Esto es para romper bucles si cambiamos el modelo
    st.session_state.modelo_anterior = None

if st.session_state.modelo_anterior != modelo_sel:
    st.session_state.modelo_anterior = modelo_sel

match modelo_sel:
    case "timegpt":
        with st.spinner(f'Ejecutando {modelo_sel}...'):
            fcst_df = timegpt.forecast_model(final_df, h, "h")
        st.write("Futuro horizon")
        st.dataframe(fcst_df.head(3))
        st.pyplot(forecast_plot(final_df, fcst_df, ["TimeGPT"]), use_container_width=True)
        st.write("Pronostico")
        st.pyplot(forecast_only_future(final_df, fcst_df, h, ["TimeGPT"], umbral), use_container_width=True)
    case "timegpt_long":
        fcst_df = timegpt_long.forecast_model(final_df, h, "h")
        st.write("Futuro horizon")
        st.dataframe(fcst_df.tail(3))
        st.pyplot(forecast_plot(final_df, fcst_df, ["TimeGPT"]), use_container_width=True)
        st.write("Pronostico")
        st.pyplot(forecast_only_future(final_df, fcst_df, h, ["TimeGPT"], umbral), use_container_width=True)
    case "auto_nhits":
        fcst_df = auto_nhits.forecast_model(final_df, h, 2, "optuna")
        st.dataframe(fcst_df.tail(3))
        st.write("Pronostico")
        st.pyplot(plot_dl_future_only(final_df, fcst_df, h, umbral))
    case "auto_tft":
        fcst_df = auto_tft.forecast_model(final_df, h, 2, "optuna")
        st.dataframe(fcst_df.tail(3))
        st.write("Pronostico")
        st.pyplot(plot_dl_future_only(final_df, fcst_df, h, umbral))

#Backtesting

if modelo_sel == "timegpt":
    d =  st.selectbox("Profundidad", ["Depth = 1", "Depth = 2", "Depth = 3", "Depth = 4", "Depth = 5"])
    p =  st.selectbox("porcentaje train: ", [70, 75, 80, 85, 90, 95, 99])
    evaluation_results, test_with_preds = evalModelGPT(final_df, p, "h", "timegpt-1")
    st.write("TRAIN/TEST")
    st.pyplot(forecast_plot_evaluation(
        train_df=final_df[:int(len(final_df) * p)],
        test_df_with_predictions=test_with_preds,
        models_to_plot=d
    ))
    st.write("ZOOM")
    st.pyplot(forecast_only_test_period(
        train_df=final_df[:int(len(final_df) * p)],
        test_df_with_predictions=test_with_preds,
        models_to_plot=d,
        umbral=umbral
    ))

