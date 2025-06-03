import ccxt
import pandas as pd

#codigo para extraer datos gracias a https://github.com/codeninja/CCXT-Historical-Data/blob/master/README.md
def retry_fetch_ohlcv(exchange, max_retries, symbol, timeframe, since, limit):
    """
    Intenta hasta `max_retries` veces obtener las velas del intercambio.
    No imprime nada. Si falla tras todos los intentos, lanza excepción.
    """
    attempts = 0
    while attempts < max_retries:
        try:
            attempts += 1
            return exchange.fetch_ohlcv(symbol, timeframe, since, limit)
        except Exception as e:
            if attempts >= max_retries:
                raise Exception(
                    f"Failed to fetch {timeframe} {symbol} OHLCV in {max_retries} attempts: {str(e)}"
                )
            # Si enableRateLimit=True, CCXT ya espera internamente.
            # Si la API es muy estricta, puedes descomentar la siguiente línea:
            # time.sleep(0.2)

def scrape_ohlcv_forward(exchange, max_retries, symbol, timeframe, since, limit):
    """
    Scrapea OHLCV hacia adelante, sin imprimir nada en consola.
    Calcula el siguiente `since` sumando el tamaño del timeframe a la última vela.
    """
    all_ohlcv = []
    from_ts = since

    # Duración del timeframe en milisegundos:
    tf_seconds = exchange.parse_timeframe(timeframe)
    tf_ms = tf_seconds * 1000

    while True:
        ohlcv_batch = retry_fetch_ohlcv(exchange, max_retries, symbol, timeframe, from_ts, limit)
        if not ohlcv_batch:
            break

        all_ohlcv.extend(ohlcv_batch)

        # Si la cantidad de velas es menor que el límite, significa que llegamos al último batch
        if len(ohlcv_batch) < limit:
            break

        # Siguiente since = timestamp del último + 1 * timeframe
        last_ts = ohlcv_batch[-1][0]
        from_ts = last_ts + tf_ms

        # Si enableRateLimit=True, CCXT meterá el delay automáticamente
        # time.sleep(0.2)

    return all_ohlcv


def scrape_ohlcv_backward(exchange, max_retries, symbol, timeframe, since, limit):
    """
    Scrapea OHLCV hacia atrás, sin imprimir nada en consola.
    Va retrocediendo hasta llegar a `since`.
    """
    # Timestamp más cercano al “ahora”
    earliest_ts = exchange.milliseconds()
    tf_seconds = exchange.parse_timeframe(timeframe)
    tf_ms = tf_seconds * 1000
    window_ms = limit * tf_ms

    all_ohlcv = []

    while True:
        fetch_since = earliest_ts - window_ms
        if fetch_since < since:
            fetch_since = since

        ohlcv_batch = retry_fetch_ohlcv(exchange, max_retries, symbol, timeframe, fetch_since, limit)
        if not ohlcv_batch:
            break

        # Si la vela más temprana del batch ya es posterior o igual al earliest, salimos
        first_ts = ohlcv_batch[0][0]
        if first_ts >= earliest_ts:
            break

        # Insertamos al inicio el nuevo batch, para mantener orden cronológico
        all_ohlcv = ohlcv_batch + all_ohlcv

        # Actualizamos earliest_ts a la vela más temprana del batch
        earliest_ts = first_ts

        # Si ya alcanzamos o pasamos `since`, interrumpimos
        if fetch_since <= since:
            break

        # Si enableRateLimit=True, CCXT meterá el delay automáticamente
        # Si quieres algún sleep manual, descomenta la siguiente línea:
        # time.sleep(0.2)

    return all_ohlcv


def scrape_candles_to_dataframe(exchange_id, symbol, timeframe, since, limit=1000, max_retries=3, direction='forward'):
    """
    Función principal: devuelve un DataFrame con todas las velas en el rango deseado.
    """
    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({'enableRateLimit': True})

    # Si `since` viene como string, lo convertimos a ms
    if isinstance(since, str):
        since = exchange.parse8601(since)

    exchange.load_markets()

    if direction == 'forward':
        raw_ohlcv = scrape_ohlcv_forward(exchange, max_retries, symbol, timeframe, since, limit)
    else:
        raw_ohlcv = scrape_ohlcv_backward(exchange, max_retries, symbol, timeframe, since, limit)

    # Convertimos a DataFrame y limpiamos duplicados
    df = pd.DataFrame(raw_ohlcv, columns=['ds', 'open', 'high', 'low', 'y', 'volume'])
    df['ds'] = pd.to_datetime(df['ds'], unit='ms')
    df.set_index('ds', inplace=True)
    df.sort_index(ascending=True, inplace=True)
    df = df[~df.index.duplicated(keep='first')]

    return df


def get_ohlcv_dataframe(exchange_id='binance', symbol='SOL/USDT', timeframe='1h', since='2025-03-01 00:00:00'):
    return scrape_candles_to_dataframe(exchange_id, symbol, timeframe, since)

def get_final_dataframe(dfv1):
    dfp = dfv1.drop(columns=["open", "high","low","volume"], axis=1)
    dfp = dfv1[['y']].reset_index().rename(columns={
    'ds': 'ds',        
    'y': 'y'      
    })
    df = dfp.copy()
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.set_index("ds")
    full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq="H")
    faltantes = full_range.difference(df.index)
    df_reindexed = df.reindex(full_range)
    df_imputado = df_reindexed.ffill()
    final_df = df_imputado.reset_index().rename(columns={"index": "ds", "y": "y"})
    final_df['unique_id'] = 'series_1' 
    return final_df

