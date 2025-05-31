import streamlit as st
import matplotlib.pyplot as plt
import os
from utils.data import get_ohlcv_dataframe, get_final_dataframe
from utils.evaluator import evaluate_forecast
from utils.plots import normal_plot, forecast_plot, forecast_only_future
from models.foundation import timegpt
st.set_page_config(layout="wide")

token = st.selectbox("Token", ["BTC/USDT", "EUR/USDT"])

@st.cache_data
def load_ohlcv_cached(symbol: str, timeframe: str, since: str):
    return get_ohlcv_dataframe("binance", symbol, timeframe, since)
@st.cache_data
def load_final_df_cached(raw_df):
    return get_final_dataframe(raw_df)
    
df = load_ohlcv_cached(token, "1h", "2010-01-01 00:00:00")
st.dataframe(df.tail(3))
final_df = get_final_dataframe(df) 
st.write("Datos procesados")
st.dataframe(final_df.tail(3))
st.pyplot(normal_plot(final_df))

MODELOS_DIR = "models" 
categorias = []
for entry in os.listdir(MODELOS_DIR):
    full_path = os.path.join(MODELOS_DIR, entry)
    if os.path.isdir(full_path):
        categorias.append(entry)

categoria_sel = st.selectbox("Categoria", sorted(categorias))
modelos_disponibles = []
cat_path = os.path.join(MODELOS_DIR, categoria_sel)
for fname in os.listdir(cat_path):
    if fname.endswith(".py") and not fname.startswith("__"):
        modelos_disponibles.append(fname[:-3])  

modelo_sel = st.selectbox("Selecciona un modelo", sorted(modelos_disponibles))
umbral = st.number_input("Umbral", 30)
h = st.number_input("Horizonte", 24)
if modelo_sel == "timegpt":
    fcst_df = timegpt.forecast_model(final_df, h, "H")
    st.write("Futuro horizon")
    st.dataframe(fcst_df.tail(3))
    st.pyplot(forecast_plot(final_df, fcst_df, ["TimeGPT"]), use_container_width=True)
    st.write("Pronostico")
    st.pyplot(forecast_only_future(final_df, fcst_df, h, ["TimeGPT"], umbral), use_container_width=True)



