import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry
import joblib
import tensorflow as tf
import numpy as np

NUM_LAGS = 13
NUM_PREVISOES = 14

# --- ALTERAÇÃO 1: Nomes das cidades atualizados para corresponder ao CSV ---
# Trocamos "Cidade_1", "Cidade_2", etc., pelos nomes reais das colunas.
# Isso torna o código mais legível e sincronizado com seus dados.
CIDADES = [
    ("vacaria_mm", -28.5122, -50.9339),
    ("guapore_mm", -28.8456, -51.8903),
    ("lagoa_vermelha_mm", -28.2086, -51.5258),
    ("passo_fundo_mm", -28.2628, -52.4067),
    ("soledade_mm", -28.8183, -52.5103),
    ("cruz_alta_mm", -28.644, -53.6063),
    ("salto_do_jacui_mm", -29.0883, -53.2125),
    ("sao_francisco_de_paula_mm", -29.4481, -50.5836),
    ("bento_goncalves_mm", -29.1714, -51.5192),
    ("caxias_do_sul_mm", -29.1681, -51.1794),
    ("lajeado_mm", -29.4669, -51.9614),
    ("taquari_mm", -29.7997, -51.8644),
    ("santa_cruz_do_sul_mm", -29.7175, -52.4258),
    ("julio_de_castilhos_mm", -29.2269, -53.6817),
    ("santa_maria_mm", -29.6842, -53.8069),
    ("viamao_mm", -30.0811, -51.0233),
    ("cachoeira_do_sul_mm", -30.0392, -52.8939),
    ("encruzilhada_do_sul_mm", -30.5439, -52.5219),
    ("cacapava_do_sul_mm", -30.5144, -53.485),
    ("sao_gabriel_mm", -30.3364, -54.32),
]

def coletar_chuva_14dias():
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    def consultar(cidade, lat, lon):
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": "rain_sum",
            "timezone": "America/Sao_Paulo",
            "forecast_days": NUM_PREVISOES
        }
        responses = openmeteo.weather_api("https://api.open-meteo.com/v1/forecast", params=params)
        response = responses[0]
        daily = response.Daily()
        valores = daily.Variables(0).ValuesAsNumpy()
        datas = pd.date_range(
            start=pd.to_datetime(daily.Time(), unit="s", utc=True),
            end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=daily.Interval()),
            inclusive="left"
        )
        # Graças à alteração na lista CIDADES, o nome da coluna aqui já será o correto (ex: "vacaria_mm")
        return pd.DataFrame({"date": datas, cidade: valores})

    dfs = [consultar(nome, lat, lon) for nome, lat, lon in CIDADES]
    df_merged = dfs[0]
    for df in dfs[1:]:
        df_merged = df_merged.merge(df, on="date")
    
    return df_merged


def prever_nivel_rio(chuvas_df: pd.DataFrame):
    # --- ALTERAÇÃO 2: Simplificação do tratamento de colunas ---
    # Agora não precisamos mais carregar 'afluentes.pkl' ou renomear colunas.
    # O código fica mais limpo e menos propenso a erros.
    
    # Carregue apenas o que é necessário para a predição
    scaler = joblib.load("models/scaler.pkl")
    model = tf.keras.models.load_model("models/lstm_model.keras")

    df_padronizado = chuvas_df.copy()

    # Identifica dinamicamente as colunas de chuva (todas, exceto 'date')
    colunas_chuva = [col for col in df_padronizado.columns if col != "date"]
    
    # AVISO: Garanta que a ordem das colunas aqui é a mesma usada para treinar o 'scaler'!
    # O 'scaler.transform' é sensível à ordem das colunas.
    df_padronizado[colunas_chuva] = scaler.transform(df_padronizado[colunas_chuva])

    entradas = []
    datas = []
    for i in range(NUM_PREVISOES - NUM_LAGS + 1):
        # Usa a lista de colunas de chuva para criar a janela de dados
        janela = df_padronizado[colunas_chuva].iloc[i:i+NUM_LAGS].values
        entradas.append(janela)
        # Pega a data correspondente ao final da janela
        datas.append(chuvas_df['date'].iloc[i + NUM_LAGS - 1])

    entradas = np.array(entradas)
    preds = model.predict(entradas).flatten()

    resultado = pd.DataFrame({
        "data": datas,
        "altura_prevista": preds
    })
    return resultado
