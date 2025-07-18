# backtest.py (CORRIGIDO E FINAL)

import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import json
import openmeteo_requests
import requests_cache
from retry_requests import retry

from src.preprocess_dataframe import preprocess_dataframe

# --- CONFIGURAÇÕES DO BACKTEST ---
ARQUIVO_NIVEIS_REAIS_CSV = 'niveis_reais_diarios.csv'
DATA_INICIO_PREVISAO = "2024-04-08"
DATA_FIM_PREVISAO = "2024-05-06"
NIVEL_INICIAL_REAL = 0.83
NUM_LAGS = 7
COTA_INUNDACAO = 3.0

CIDADES = {
    "vacaria_mm": (-28.5108, -50.9389), "guapore_mm": (-28.8489, -51.8903),
    "lagoa_vermelha_mm": (-28.2086, -51.5258), "passo_fundo_mm": (-28.2628, -52.4067),
    "soledade_mm": (-28.8189, -52.5103), "cruz_alta_mm": (-28.6389, -53.6064),
    "salto_do_jacui_mm": (-29.0950, -53.2144), "sao_francisco_de_paula_mm": (-29.4481, -50.5822),
    "bento_goncalves_mm": (-29.1686, -51.5189), "caxias_do_sul_mm": (-29.1678, -51.1789),
    "lajeado_mm": (-29.4669, -51.9614), "taquari_mm": (-29.7983, -51.8619),
    "santa_cruz_do_sul_mm": (-29.7178, -52.4258), "julio_de_castilhos_mm": (-29.2269, -53.6817),
    "santa_maria_mm": (-29.6842, -53.8069), "viamao_mm": (-30.0811, -51.0233),
    "cachoeira_do_sul_mm": (-30.0392, -52.8939), "encruzilhada_do_sul_mm": (-30.5428, -52.5219),
    "cacapava_do_sul_mm": (-30.5128, -53.4911), "sao_gabriel_mm": (-30.3358, -54.3200)
}

def fetch_rain_data(start_date_str, end_date_str):
    print(f"⏳ Buscando dados de chuva de {start_date_str} a {end_date_str}...")
    openmeteo = openmeteo_requests.Client(session=retry(requests_cache.CachedSession('.cache', expire_after=3600), retries=5, backoff_factor=0.2))
    url = "https://archive-api.open-meteo.com/v1/archive"
    dfs = []
    for nome_cidade, (lat, lon) in CIDADES.items():
        params = {
            "latitude": lat, "longitude": lon,
            "start_date": start_date_str, "end_date": end_date_str,
            "daily": "rain_sum", "timezone": "America/Sao_Paulo"
        }
        try:
            responses = openmeteo.weather_api(url, params=params)
            response = responses[0]
            daily = response.Daily()
            rain_values = daily.Variables(0).ValuesAsNumpy()
            df = pd.DataFrame(rain_values, columns=[nome_cidade])
            dfs.append(df)
        except Exception as e:
            print(f"❌ Falha ao buscar {nome_cidade}: {e}")
            dias = (pd.to_datetime(end_date_str) - pd.to_datetime(start_date_str)).days + 1
            dfs.append(pd.DataFrame(np.zeros(dias), columns=[nome_cidade]))

    chuva_df = pd.concat(dfs, axis=1)
    chuva_df.index = pd.date_range(start=start_date_str, end=end_date_str, freq='D')
    print("✅ Dados de chuva coletados.")
    return chuva_df

def plotar_comparacao(df_resultado):
    os.makedirs('results', exist_ok=True)
    caminho_saida = f"results/backtest_{DATA_INICIO_PREVISAO}_a_{DATA_FIM_PREVISAO}.png"
    print(f"📈 Gerando gráfico de comparação em {caminho_saida}...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(18, 9))
    ax.plot(df_resultado.index, df_resultado['nivel_real'], 'o-', label='Nível Real', color='black', linewidth=2.5)
    ax.plot(df_resultado.index, df_resultado['nivel_previsto'], 'o--', label='Previsão do Modelo', color='crimson', linewidth=2)
    ax.axhline(y=COTA_INUNDACAO, color='darkblue', linestyle=':', label=f'Cota de Inundação ({COTA_INUNDACAO:.2f} m)')
    ax.set_title("Backtest do Modelo", fontsize=18, weight='bold')
    ax.set_xlabel("Data", fontsize=12); ax.set_ylabel("Nível do Rio (m)", fontsize=12)
    ax.legend(fontsize=12); ax.grid(True, linestyle='--', linewidth=0.5)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout(); plt.savefig(caminho_saida, dpi=150); plt.close()
    print("✅ Gráfico salvo.")

def run_backtest():
    print("\n--- INICIANDO BACKTEST ---")

    # --- CORREÇÃO: Carregar os arquivos com os nomes corretos ---
    # O seu train.py cria 'lstm_model_absolute.keras' e 'scaler_alvo.pkl'.
    try:
        print("✅ Carregando modelo e scalers do treino...")
        model = tf.keras.models.load_model('models/lstm_model_absolute.keras')
        scaler_saida = joblib.load('models/scaler_alvo.pkl')
        scaler_entradas = joblib.load('models/scaler_entradas.pkl')
        IS_DELTA_MODEL = False # O modelo é absoluto, não delta
    except (IOError, ValueError) as e:
        print(f"❌ Erro fatal: Não foi possível carregar os arquivos de modelo da pasta 'models/'.")
        print(f"   Certifique-se de que o 'train.py' foi executado com sucesso. Erro: {e}")
        return

    COLUNA_NIVEL_ABSOLUTO = 'altura_rio_guaiba_m'
    
    # O resto da lógica continua igual...
    DIAS_DE_FOLGA = 15
    data_inicio = pd.to_datetime(DATA_INICIO_PREVISAO)
    data_fim = pd.to_datetime(DATA_FIM_PREVISAO)
    data_inicio_hist = data_inicio - pd.Timedelta(days=NUM_LAGS + DIAS_DE_FOLGA)

    df_chuva_bruto = fetch_rain_data(data_inicio_hist.strftime('%Y-%m-%d'), data_fim.strftime('%Y-%m-%d'))
    df_chuva_bruto[COLUNA_NIVEL_ABSOLUTO] = NIVEL_INICIAL_REAL
    
    df_processado = preprocess_dataframe(df_chuva_bruto, coluna_nivel=COLUNA_NIVEL_ABSOLUTO)
    df_processado['data'] = pd.to_datetime(df_processado['data'])
    df_processado.set_index('data', inplace=True)

    FEATURES_ENTRADA = [col for col in df_processado.columns if col != COLUNA_NIVEL_ABSOLUTO]
    print(f"ℹ️  {len(FEATURES_ENTRADA)} features foram geradas dinamicamente para o backtest.")
    
    previsoes = []
    nivel_anterior = NIVEL_INICIAL_REAL
    
    print("🔮 Simulando previsão dia a dia com atualização dinâmica...")
    for data_previsao in pd.date_range(start=data_inicio, end=data_fim):
        fim_janela = data_previsao - pd.Timedelta(days=1)
        inicio_janela = fim_janela - pd.Timedelta(days=NUM_LAGS - 1)
        
        try:
            historico_janela = df_processado.loc[inicio_janela:fim_janela]
            
            if len(historico_janela) < NUM_LAGS:
                print(f"⚠️ Janela de dados insuficiente em {data_previsao.date()}, repetindo previsão anterior.")
                previsoes.append(nivel_anterior)
                continue
            
            X = historico_janela[FEATURES_ENTRADA]
        except KeyError:
            print(f"❌ Datas não encontradas para o dia {data_previsao.date()}. Repetindo previsão anterior.")
            previsoes.append(nivel_anterior)
            continue
            
        X_scaled = scaler_entradas.transform(X)
        X_input = np.expand_dims(X_scaled, axis=0)
        
        saida_scaled = model.predict(X_input, verbose=0)[0][0]
        saida_real = scaler_saida.inverse_transform([[saida_scaled]])[0][0]
        
        # Como o modelo é absoluto, a previsão é o próprio valor
        nivel_previsto = saida_real
        
        previsoes.append(nivel_previsto)
        nivel_anterior = nivel_previsto

    df_reais = pd.read_csv(ARQUIVO_NIVEIS_REAIS_CSV, parse_dates=['data'], index_col='data')
    df_resultado = df_reais.loc[DATA_INICIO_PREVISAO:DATA_FIM_PREVISAO].copy()
    
    if len(previsoes) != len(df_resultado):
        print(f"❌ Erro: O número de previsões ({len(previsoes)}) não bate com o número de dias reais ({len(df_resultado)}).")
        return

    df_resultado['nivel_previsto'] = previsoes
    df_resultado.rename(columns={'altura_rio_guaiba_m': 'nivel_real'}, inplace=True)
    df_resultado.dropna(inplace=True)

    if df_resultado.empty:
        print("❌ Nenhum resultado para calcular métricas após remover NaNs.")
        return

    mae = mean_absolute_error(df_resultado['nivel_real'], df_resultado['nivel_previsto'])
    rmse = mean_squared_error(df_resultado['nivel_real'], df_resultado['nivel_previsto']) ** 0.5

    print("\n--- MÉTRICAS DO MODELO ---")
    print(f"MAE  = {mae:.3f} m")
    print(f"RMSE = {rmse:.3f} m")
    print("---------------------------")
    print("\n--- RESULTADO DIA A DIA ---")
    df_print = df_resultado.copy()
    df_print['nivel_real'] = df_print['nivel_real'].round(2)
    df_print['nivel_previsto'] = df_print['nivel_previsto'].round(2)
    print(df_print.to_string())
    print("---------------------------")

    plotar_comparacao(df_resultado)

if __name__ == "__main__":
    run_backtest()