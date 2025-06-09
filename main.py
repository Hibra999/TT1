import streamlit as st
import matplotlib.pyplot as plt
import os
import sys
import asyncio
if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
from utils.data import get_ohlcv_dataframe, get_final_dataframe
from utils.plots import normal_plot, forecast_plot, forecast_only_future, plot_dl, plot_dl_future_only, forecast_plot_evaluation, forecast_only_test_period, plot_backtestGPT
from models.foundation import timegpt, timegpt_long
from models.dl import auto_nhits, auto_tft
from models.stats import auto_arima
from utils.evaluator import timeGPTrico, maeMSEetc
st.set_page_config(layout="wide")

HORIZONTES = {
    "Hace 1 hora":   1,
    "Hace 24 horas": 24,
    "Hace 1 semana": 168,
    "Hace 1 mes":    730
}
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

umbral = st.number_input("Umbral", 30)
st.write("Hora, Dia, Semana, Mes")
h =  st.selectbox("Horizonte, hrs", [1, 24, 168, 730])
modelo_sel = st.selectbox("Selecciona un modelo", modelos_disponibles) 
boton1 = st.button("Ejecutar modelo") #BOTON LOCO

if boton1:
    with st.spinner(f'Ejecutando {modelo_sel}'):
        if modelo_sel == "timegpt":
            fcst_df = timegpt.forecast_model(final_df, h, "h")
            st.write("Futuro horizon")
            st.dataframe(fcst_df.head(3))
            st.pyplot(forecast_plot(final_df, fcst_df, ["TimeGPT"]), use_container_width=True)
            st.write("Pronostico")
            st.pyplot(forecast_only_future(final_df, fcst_df, h, ["TimeGPT"], umbral), use_container_width=True)
        elif modelo_sel == "timegpt_long":
            fcst_df = timegpt_long.forecast_model(final_df, h, "h")
            st.write("Futuro horizon")
            st.dataframe(fcst_df.tail(3))
            st.pyplot(forecast_plot(final_df, fcst_df, ["TimeGPT"]), use_container_width=True)
            st.write("Pronostico")
            st.pyplot(forecast_only_future(final_df, fcst_df, h, ["TimeGPT"], umbral), use_container_width=True)
        elif modelo_sel == "auto_nhits":
            fcst_df = auto_nhits.forecast_model(final_df, h, 2, "optuna")
            st.dataframe(fcst_df.tail(3))
            st.write("Pronostico")
            st.pyplot(plot_dl_future_only(final_df, fcst_df, h, umbral))
        elif modelo_sel == "auto_tft":
            fcst_df = auto_tft.forecast_model(final_df, h, 2, "optuna")
            st.dataframe(fcst_df.tail(3))
            st.write("Pronostico")
            st.pyplot(plot_dl_future_only(final_df, fcst_df, h, umbral))
        elif modelo_sel == "auto_arima":
            fcst_df = auto_arima.forecast_model(final_df, h, "h", [90])
            st.dataframe(fcst_df.tail(3))
            st.pyplot(plot_dl_future_only(final_df, fcst_df, h, umbral))
    
#Backtesting
st.write("Backstesting")
if modelo_sel == "timegpt":
    h_label = st.selectbox("Horizonte", list(HORIZONTES.keys()))
    p = HORIZONTES[h_label]
    boton3 =  st.button("Ejecutar")
    if boton3:
        fcst_new = timeGPTrico(final_df, p, "h", "timegpt-1")
        st.pyplot(plot_backtestGPT(fcst_new["train_data"], fcst_new["fcst"]))
        eval_df = maeMSEetc(fcst_new["train_data"], fcst_new["fcst"])
        st.dataframe(eval_df)
elif modelo_sel == "timegpt_long":
    h_label = st.selectbox("Horizonte", list(HORIZONTES.keys()))
    p = HORIZONTES[h_label]
    boton3 =  st.button("Ejecutar")
    if boton3:
        fcst_new = timeGPTrico(final_df, p, "h", "timegpt-1")
        st.pyplot(plot_backtestGPT(fcst_new["train_data"], fcst_new["fcst"]))
        eval_df = maeMSEetc(fcst_new["train_data"], fcst_new["fcst"])
        st.dataframe(eval_df)

