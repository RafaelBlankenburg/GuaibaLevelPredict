# Arquivo: backtest.py

import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import sys
import time
import json
import openmeteo_requests
import requests_cache
from retry_requests import retry

# --- 1. CONFIGURAÇÃO DO TESTE ---
ARQUIVO_NIVEIS_REAIS_CSV = 'niveis_reais_diarios.csv'
DATA_INICIO_PREVISAO = "2024-04-08"
DATA_FIM_PREVISAO = "2024-05-06"
NIVEL_INICIAL_REAL = 0.83 
NUM_LAGS = 7

# --- 2. CONSTANTES DO PROJETO ---
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

def plotar_comparacao(df_resultado):
    caminho_saida = f"results/backtest_{DATA_INICIO_PREVISAO}_a_{DATA_FIM_PREVISAO}.png"
    print(f"📈 Gerando gráfico de comparação em {caminho_saida}...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(18, 9))
    ax.plot(df_resultado.index, df_resultado['nivel_real'], 'o-', label='Nível Real', color='black', linewidth=2.5, zorder=5)
    ax.plot(df_resultado.index, df_resultado['nivel_previsto'], 'o--', label='Previsão do Modelo', color='crimson', linewidth=2, zorder=5)
    ax.axhline(y=COTA_INUNDACAO, color='darkblue', linestyle=':', label=f'Cota de Inundação ({COTA_INUNDACAO:.2f} m)')
    ax.set_title(f"Backtest do Modelo: {df_resultado.index[0].strftime('%d/%m/%Y')} a {df_resultado.index[-1].strftime('%d/%m/%Y')}", fontsize=18, weight='bold')
    ax.set_xlabel('Data', fontsize=12); ax.set_ylabel('Nível do Rio (metros)', fontsize=12)
    ax.legend(fontsize=12); ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m')); plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    fig.tight_layout()
    os.makedirs('results', exist_ok=True); plt.savefig(caminho_saida, dpi=150); plt.close(fig)
    print("✅ Gráfico de comparação salvo.")

def fetch_rain_data(start_date_str, end_date_str):
    print(f"⏳ Buscando dados de chuva de {start_date_str} a {end_date_str}...")
    openmeteo = openmeteo_requests.Client(session = retry(requests_cache.CachedSession('.cache', expire_after=3600), retries=5, backoff_factor=0.2))
    url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
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
            print(f"❌ Falha ao buscar dados para {nome_cidade}. Erro: {e}. Preenchendo com 0.")
            num_dias = (pd.to_datetime(end_date_str) - pd.to_datetime(start_date_str)).days + 1
            dfs.append(pd.DataFrame([0.0] * num_dias, columns=[nome_cidade]))
    chuva_df = pd.concat(dfs, axis=1)
    all_dates = pd.date_range(start=start_date_str, end=end_date_str, freq='D')
    chuva_df.index = all_dates
    print("✅ Dados de chuva coletados com sucesso.")
    return chuva_df

def run_backtest():
    print("--- INICIANDO BACKTESTING DO MODELO ---")
    model = tf.keras.models.load_model('models/lstm_model_delta.keras')
    scaler_entradas = joblib.load('models/scaler_entradas.pkl')
    scaler_delta = joblib.load('models/scaler_delta.pkl')
    with open('models/training_columns.json', 'r') as f: config_colunas = json.load(f)
    FEATURES_ENTRADA = config_colunas['features_entrada']
    COLUNA_NIVEL_ABSOLUTO = config_colunas['coluna_nivel_absoluto']
    
    data_inicio = pd.to_datetime(DATA_INICIO_PREVISAO)
    data_fim = pd.to_datetime(DATA_FIM_PREVISAO)
    data_inicio_hist_chuva = data_inicio - pd.Timedelta(days=NUM_LAGS)
    
    df_chuva_total = fetch_rain_data(data_inicio_hist_chuva.strftime('%Y-%m-%d'), data_fim.strftime('%Y-%m-%d'))
    
    # Prepara a janela de histórico inicial apenas com os dados brutos
    data_ponto_partida = data_inicio - pd.Timedelta(days=1)
    historico_janela = df_chuva_total.loc[:data_ponto_partida].copy()
    historico_janela[COLUNA_NIVEL_ABSOLUTO] = NIVEL_INICIAL_REAL # Preenche o histórico com o nível inicial
    
    previsoes = []
    print("🔮 Simulando previsão dia a dia...")
    for data_previsao in pd.date_range(start=data_inicio, end=data_fim):
        # --- LÓGICA CORRIGIDA: ENGENHARIA DE FEATURES DENTRO DO LOOP ---
        janela_com_eng = historico_janela.copy()
        
        # Recalcula as features de engenharia para a janela atual
        cidades_chuva = list(CIDADES.keys())
        for cidade in cidades_chuva:
            janela_com_eng[f'delta_{cidade}'] = janela_com_eng[cidade].diff().fillna(0)
            janela_com_eng[f'acum_{cidade}_3d'] = janela_com_eng[cidade].rolling(window=3).sum().fillna(0)
        
        janela_para_prever = janela_com_eng[FEATURES_ENTRADA]
        # --- FIM DA CORREÇÃO ---

        janela_padronizada = scaler_entradas.transform(janela_para_prever)
        janela_lstm_input = np.expand_dims(janela_padronizada, axis=0)
        
        delta_previsto_scaled = model.predict(janela_lstm_input, verbose=0)[0][0]
        delta_previsto_real = scaler_delta.inverse_transform([[delta_previsto_scaled]])[0][0]
        
        nivel_anterior = historico_janela.iloc[-1][COLUNA_NIVEL_ABSOLUTO]
        nivel_previsto = nivel_anterior + delta_previsto_real
        previsoes.append(nivel_previsto)
        
        # Atualiza a janela de histórico para a próxima iteração
        proxima_linha_chuva = df_chuva_total.loc[data_previsao].copy()
        proxima_linha_df = pd.DataFrame([proxima_linha_chuva], index=[data_previsao])
        proxima_linha_df[COLUNA_NIVEL_ABSOLUTO] = nivel_previsto
        
        historico_janela = pd.concat([historico_janela.iloc[1:], proxima_linha_df])

    df_niveis_reais = pd.read_csv(ARQUIVO_NIVEIS_REAIS_CSV, parse_dates=['data'], index_col='data')
    df_resultado = df_niveis_reais.loc[DATA_INICIO_PREVISAO:DATA_FIM_PREVISAO].copy()
    df_resultado['nivel_previsto'] = previsoes
    df_resultado.rename(columns={'altura_rio_guaiba_m': 'nivel_real'}, inplace=True)
    
    mae = mean_absolute_error(df_resultado['nivel_real'], df_resultado['nivel_previsto'])
    rmse = root_mean_squared_error(df_resultado['nivel_real'], df_resultado['nivel_previsto'])

    print("\n--- MÉTRICAS DE PERFORMANCE DO BACKTEST ---")
    print(f"Erro Médio Absoluto (MAE): {mae:.3f} metros (ou {(mae*100):.1f} cm em média)")
    print(f"Raiz do Erro Quadrático Médio (RMSE): {rmse:.3f} metros")
    print("------------------------------------------")

    plotar_comparacao(df_resultado)


if __name__ == "__main__":
    run_backtest()