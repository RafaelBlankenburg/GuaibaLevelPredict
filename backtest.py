# backtest.py (VERSÃO FINAL COMPLETA)

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

# --- CONFIGURAções DO BACKTEST ---
ARQUIVO_NIVEIS_REAIS_CSV = 'data/niveis_reais_diarios.csv'
DATA_INICIO_PREVISAO = "2024-04-08"
DATA_FIM_PREVISAO = "2024-05-06"
NIVEL_INICIAL_REAL = 0.83

# --- CONSTANTES ALINHADAS COM O TREINO ---
NUM_LAGS = 6
DIAS_ROLLING_MAX = 7
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
        params = {"latitude": lat, "longitude": lon, "start_date": start_date_str, "end_date": end_date_str, "daily": "rain_sum", "timezone": "America/Sao_Paulo"}
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
    ax.set_title("Backtest do Modelo - Lógica Final", fontsize=18, weight='bold')
    ax.set_xlabel("Data", fontsize=12); ax.set_ylabel("Nível do Rio (m)", fontsize=12)
    ax.legend(fontsize=12); ax.grid(True, linestyle='--', linewidth=0.5)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout(); plt.savefig(caminho_saida, dpi=150); plt.close()
    print("✅ Gráfico salvo.")

def run_backtest():
    print(f"\n--- INICIANDO BACKTEST 'HONESTO' (LÓGICA FINAL) ---")
    try:
        model = tf.keras.models.load_model('models/lstm_model_delta.keras')
        scaler_saida = joblib.load('models/scaler_delta.pkl')
        scaler_entradas = joblib.load('models/scaler_entradas.pkl')
        with open('models/training_columns.json', 'r') as f:
            FEATURES_ENTRADA = json.load(f)['features_entrada']
    except Exception as e:
        print(f"❌ Erro fatal ao carregar arquivos do modelo: {e}. Execute o train.py primeiro.")
        return

    COLUNA_NIVEL_ABSOLUTO = 'altura_rio_guaiba_m'
    data_inicio_previsao = pd.to_datetime(DATA_INICIO_PREVISAO)
    data_fim_previsao = pd.to_datetime(DATA_FIM_PREVISAO)
    
    # 1. PREPARAÇÃO DO PONTO DE PARTIDA "FRIO" (REALISTA)
    DIAS_HISTORico_NECESSARIO = NUM_LAGS + DIAS_ROLLING_MAX + 5
    data_fim_historico = data_inicio_previsao - pd.Timedelta(days=1)
    data_inicio_historico = data_fim_historico - pd.Timedelta(days=DIAS_HISTORico_NECESSARIO)
    
    df_chuva_historica = fetch_rain_data(data_inicio_historico.strftime('%Y-%m-%d'), data_fim_historico.strftime('%Y-%m-%d'))
    
    df_historico_bruto = df_chuva_historica.copy()
    df_historico_bruto[COLUNA_NIVEL_ABSOLUTO] = NIVEL_INICIAL_REAL

    df_historico_processado = preprocess_dataframe(df_historico_bruto, coluna_nivel=COLUNA_NIVEL_ABSOLUTO)
    janela_atual = df_historico_processado[FEATURES_ENTRADA].tail(NUM_LAGS)

    # 2. PREPARAÇÃO DOS DADOS "FUTUROS" (CHUVA REAL DO PERÍODO DE TESTE)
    df_chuva_futura = fetch_rain_data(DATA_INICIO_PREVISAO, DATA_FIM_PREVISAO)
    
    # 3. INÍCIO DO LOOP DE PREVISÃO
    previsoes = []
    nivel_anterior = NIVEL_INICIAL_REAL
    historico_deltas = [0.0] * 3
    dias_simulados = 0
    df_simulacao_bruto = df_historico_bruto.copy()

    print("🔮 Simulando previsão dia a dia...")
    for data_previsao in pd.date_range(start=data_inicio_previsao, end=data_fim_previsao):
        dias_simulados += 1
        X = janela_atual
        X_scaled = scaler_entradas.transform(X)
        X_input = np.expand_dims(X_scaled, axis=0)
        
        delta_scaled = model.predict(X_input, verbose=0)[0][0]
        delta_bruto = scaler_saida.inverse_transform([[delta_scaled]])[0][0]
        
        if dias_simulados <= 2:
            delta_bruto *= 0.7 

        historico_deltas.pop(0)
        historico_deltas.append(delta_bruto)
        delta_suavizado = np.mean(historico_deltas[-2:])
        
        chuva_recente_taquari = janela_atual['bomba_chuva_taquari_3d'].iloc[-1]
        if delta_suavizado > 0.02 and historico_deltas[-2] > 0.01 and chuva_recente_taquari > 30:
            acelerador = 1 + (delta_suavizado * 0.5) 
            delta_final = delta_suavizado * acelerador
        else:
            delta_final = delta_suavizado
        
        nivel_previsto = nivel_anterior + delta_final
        nivel_previsto = max(0, nivel_previsto)
        previsoes.append(nivel_previsto)
        nivel_anterior = nivel_previsto

        # ATUALIZAÇÃO PARA O PRÓXIMO CICLO
        chuva_do_dia = df_chuva_futura.loc[[data_previsao]]
        chuva_do_dia[COLUNA_NIVEL_ABSOLUTO] = nivel_previsto
        df_simulacao_bruto = pd.concat([df_simulacao_bruto, chuva_do_dia])
        df_temp_processado = preprocess_dataframe(df_simulacao_bruto.tail(DIAS_HISTORico_NECESSARIO), coluna_nivel=COLUNA_NIVEL_ABSOLUTO)
        proxima_linha_features = df_temp_processado[FEATURES_ENTRADA].tail(1)
        janela_atual = pd.concat([janela_atual.iloc[1:], proxima_linha_features])

    # --- Bloco de comparação (carrega o CSV de níveis reais apenas no final) ---
    df_niveis_reais = pd.read_csv(ARQUIVO_NIVEIS_REAIS_CSV, parse_dates=['data'], index_col='data')
    df_resultado = df_niveis_reais.loc[DATA_INICIO_PREVISAO:DATA_FIM_PREVISAO].copy()
    df_resultado = df_resultado.iloc[:len(previsoes)]
    df_resultado['nivel_previsto'] = previsoes
    df_resultado.rename(columns={'altura_rio_guaiba_m': 'nivel_real'}, inplace=True)
    df_resultado.dropna(inplace=True)
    
    mae = mean_absolute_error(df_resultado['nivel_real'], df_resultado['nivel_previsto'])
    rmse = mean_squared_error(df_resultado['nivel_real'], df_resultado['nivel_previsto']) ** 0.5
    
    print("\n--- MÉTRICAS DO MODELO (BACKTEST HONESTO) ---")
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