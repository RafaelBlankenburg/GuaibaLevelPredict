# main.py (VERSÃO FINAL E "HONESTA")

import pandas as pd
import numpy as np
import os
import json
import joblib
import tensorflow as tf

from src.data_collection import coletar_dados_chuva, coletar_nivel_atual_rio
from src.visualization import gerar_grafico_previsao
from src.preprocess_dataframe import preprocess_dataframe

# --- CONFIGURAÇÕES GERAIS ---
NUM_LAGS_MODELO = 6
DIAS_ROLLING_MAX = 7

NUM_DIAS_PREVISAO_CHUVA = 14
DIAS_ADICIONAIS_ESTIMATIVA = 10
DIAS_TOTAIS_PREVISAO = NUM_DIAS_PREVISAO_CHUVA + DIAS_ADICIONAIS_ESTIMATIVA
COTA_INUNDACAO = 3.0
NIVEL_MINIMO_ESTIAGEM = 0.6

def run_prediction_scenarios():
    print("--- INICIANDO ROTINA DE PREVISÃO (LÓGICA 'HONESTA') ---")
    os.makedirs('results', exist_ok=True)

    try:
        model = tf.keras.models.load_model('models/lstm_model_delta.keras')
        scaler_saida = joblib.load('models/scaler_delta.pkl')
        scaler_entradas = joblib.load('models/scaler_entradas.pkl')
        with open('models/training_columns.json', 'r') as f:
            FEATURES_ENTRADA = json.load(f)['features_entrada']
        print("✅ Modelo e scalers DELTA carregados com sucesso.")
    except Exception as e:
        print(f"❌ Erro fatal ao carregar arquivos do modelo: {e}. Execute o train.py primeiro.")
        return

    # --- 1. PREPARAÇÃO DO PONTO DE PARTIDA REALISTA ---
    COLUNA_NIVEL_ABSOLUTO = 'altura_rio_guaiba_m'
    hoje = pd.to_datetime('today').normalize()
    nivel_atual = coletar_nivel_atual_rio()

    # Busca APENAS o histórico de chuva necessário para a primeira janela
    DIAS_HISTORICO_NECESSARIO = NUM_LAGS_MODELO + DIAS_ROLLING_MAX + 5
    df_chuva_historica = coletar_dados_chuva(dias_historico=DIAS_HISTORICO_NECESSARIO, dias_previsao=0)

    # Cria um histórico de níveis assumindo que o nível era constante no valor atual
    df_historico_bruto = df_chuva_historica.copy()
    df_historico_bruto[COLUNA_NIVEL_ABSOLUTO] = nivel_atual

    # Processa este histórico para obter a janela inicial de features
    df_historico_processado = preprocess_dataframe(df_historico_bruto, coluna_nivel=COLUNA_NIVEL_ABSOLUTO)
    janela_atual = df_historico_processado[FEATURES_ENTRADA].tail(NUM_LAGS_MODELO)

    # --- 2. PREPARAÇÃO DOS DADOS FUTUROS (PREVISÃO DE CHUVA) ---
    print("\n--- Cenário 1: PREVISÃO COM ESTIAGEM ---")
    df_chuva_previsao = coletar_dados_chuva(dias_historico=0, dias_previsao=NUM_DIAS_PREVISAO_CHUVA)

    if DIAS_ADICIONAIS_ESTIMATIVA > 0:
        datas_futuras = pd.date_range(start=df_chuva_previsao.index[-1] + pd.Timedelta(days=1), periods=DIAS_ADICIONAIS_ESTIMATIVA, freq='D')
        df_zeros = pd.DataFrame(0, index=datas_futuras, columns=[col for col in df_chuva_previsao.columns if col.endswith('_mm')])
        df_chuva_previsao = pd.concat([df_chuva_previsao, df_zeros])

    # --- 3. LOOP DE PREVISÃO DINÂMICA ---
    previsoes = []
    nivel_anterior = nivel_atual
    historico_deltas = [0.0] * 3
    df_simulacao_bruto = df_historico_bruto.copy()

    print("🔮 Simulando previsão dia a dia...")
    for data_previsao in pd.date_range(start=hoje, periods=DIAS_TOTAIS_PREVISAO):
        X = janela_atual
        X_scaled = scaler_entradas.transform(X)
        X_input = np.expand_dims(X_scaled, axis=0)

        delta_scaled = model.predict(X_input, verbose=0)[0][0]
        delta_bruto = scaler_saida.inverse_transform([[delta_scaled]])[0][0]

        # Lógica de Suavização e Aceleração
        historico_deltas.pop(0)
        historico_deltas.append(delta_bruto)
        delta_suavizado = np.mean(historico_deltas[-2:])

        if delta_suavizado > 0.01 and historico_deltas[-2] > 0.01:
            acelerador = 1 + (delta_suavizado * 0.5)
            delta_final = delta_suavizado * acelerador
        else:
            delta_final = delta_suavizado

        nivel_previsto = nivel_anterior + delta_final
        nivel_previsto = max(NIVEL_MINIMO_ESTIAGEM, nivel_previsto)
        previsoes.append({'data': data_previsao, 'nivel_m': nivel_previsto})

        nivel_anterior = nivel_previsto

        # ATUALIZAÇÃO PARA O PRÓXIMO CICLO
        chuva_do_dia = df_chuva_previsao.loc[[data_previsao]]
        chuva_do_dia[COLUNA_NIVEL_ABSOLUTO] = nivel_previsto
        
        df_simulacao_bruto = pd.concat([df_simulacao_bruto, chuva_do_dia])
        
        df_temp_processado = preprocess_dataframe(df_simulacao_bruto.tail(DIAS_HISTORICO_NECESSARIO), coluna_nivel=COLUNA_NIVEL_ABSOLUTO)
        proxima_linha_features = df_temp_processado[FEATURES_ENTRADA].tail(1)

        janela_atual = pd.concat([janela_atual.iloc[1:], proxima_linha_features])

    df_previsao_final = pd.DataFrame(previsoes)

    # --- Bloco de plotagem e resultados ---
    # (O resto do código continua igual)
    gerar_grafico_previsao(
        df_previsao=df_previsao_final,
        ponto_de_corte=NUM_DIAS_PREVISAO_CHUVA,
        cota_inundacao=COTA_INUNDACAO,
        caminho_saida='results/previsao_nivel_rio.png'
    )

    df_previsao_texto = df_previsao_final.copy()
    df_previsao_texto['nivel_m'] = df_previsao_texto['nivel_m'].round(2)
    df_previsao_texto['data'] = pd.to_datetime(df_previsao_texto['data']).dt.strftime('%d/%m/%Y')

    print(f"\n📈 Previsões para {NUM_DIAS_PREVISAO_CHUVA} dias e Estimativas para mais {DIAS_ADICIONAIS_ESTIMATIVA} dias:\n")
    print(df_previsao_texto.to_string(index=False))

    df_previsao_texto.to_csv("results/previsao_nivel_rio_com_estimativa.csv", index=False)
    print("\n✅ Previsões salvas em results/previsao_nivel_rio_com_estimativa.csv")


if __name__ == "__main__":
    run_prediction_scenarios()