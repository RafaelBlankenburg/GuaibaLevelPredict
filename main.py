# main.py (VERSÃO FINAL, SIMPLES E ROBUSTA)

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
NUM_LAGS_MODELO = 7
DIAS_ROLLING_MAX = 7 
NUM_DIAS_HISTORICO = NUM_LAGS_MODELO + DIAS_ROLLING_MAX + 5

NUM_DIAS_PREVISAO_CHUVA = 14
DIAS_ADICIONAIS_ESTIMATIVA = 10
DIAS_TOTAIS_PREVISAO = NUM_DIAS_PREVISAO_CHUVA + DIAS_ADICIONAIS_ESTIMATIVA
COTA_INUNDACAO = 3.0

def gerar_janelas_para_previsao(df_features, num_lags):
    """Gera todas as janelas de dados necessárias para a previsão de uma vez."""
    X = []
    # Começamos do primeiro dia que tem um histórico completo de 'num_lags' para trás
    for i in range(num_lags, len(df_features)):
        janela_x = df_features.iloc[i - num_lags : i].values
        X.append(janela_x)
    return np.array(X)

def run_prediction_scenarios():
    print("--- INICIANDO ROTINA DE PREVISÃO (LÓGICA DIRETA) ---")
    os.makedirs('results', exist_ok=True) 

    try:
        model = tf.keras.models.load_model('models/lstm_model_delta.keras')
        scaler_saida = joblib.load('models/scaler_delta.pkl')
        scaler_entradas = joblib.load('models/scaler_entradas.pkl')
        with open('models/training_columns.json', 'r') as f:
            FEATURES_ENTRADA = json.load(f)['features_entrada']
        print("✅ Modelo e scalers DELTA carregados com sucesso.")
    except Exception as e:
        print(f"❌ Erro ao carregar os arquivos do modelo: {e}")
        return

    # 1. COLETA DE DADOS
    df_chuva_total = coletar_dados_chuva(NUM_DIAS_HISTORICO, NUM_DIAS_PREVISAO_CHUVA)
    nivel_atual = coletar_nivel_atual_rio()
    
    # 2. PREPARAÇÃO DO CENÁRIO DE PREVISÃO
    df_chuva_cenario1 = df_chuva_total.copy()
    if DIAS_ADICIONAIS_ESTIMATIVA > 0:
        datas_futuras = pd.date_range(start=df_chuva_cenario1.index[-1] + pd.Timedelta(days=1), periods=DIAS_ADICIONAIS_ESTIMATIVA, freq='D')
        df_zeros = pd.DataFrame(0, index=datas_futuras, columns=[col for col in df_chuva_cenario1.columns if col.endswith('_mm')])
        df_chuva_cenario1 = pd.concat([df_chuva_cenario1, df_zeros])

    # 3. PROCESSAMENTO ÚNICO DE FEATURES
    df_chuva_cenario1['altura_rio_guaiba_m'] = 0 
    df_features_processadas = preprocess_dataframe(df_chuva_cenario1, coluna_nivel='altura_rio_guaiba_m')
    df_features_processadas = df_features_processadas[FEATURES_ENTRADA]
    
    # 4. PREVISÃO DIRETA (SEM LOOP AUTO-REGRESSIVO)
    print("🔮 Gerando janelas de dados e prevendo todas as variações (deltas)...")
    
    # Prepara todas as janelas de input de uma vez
    X_previsao = gerar_janelas_para_previsao(df_features_processadas, NUM_LAGS_MODELO)
    X_previsao_scaled = scaler_entradas.transform(X_previsao.reshape(-1, X_previsao.shape[2])).reshape(X_previsao.shape)
    
    # O modelo prevê todos os deltas de uma vez
    deltas_scaled = model.predict(X_previsao_scaled)
    deltas_previstos = scaler_saida.inverse_transform(deltas_scaled).flatten()
    
    # 5. CÁLCULO DA CURVA FINAL DO NÍVEL DO RIO
    niveis_previstos = []
    nivel_anterior = nivel_atual
    NIVEL_MINIMO_ESTIAGEM = 0.6
    
    for delta in deltas_previstos:
        nivel_previsto = nivel_anterior + delta
        nivel_previsto = max(NIVEL_MINIMO_ESTIAGEM, nivel_previsto)
        niveis_previstos.append(nivel_previsto)
        nivel_anterior = nivel_previsto

    # 6. GERAÇÃO DE GRÁFICOS E RESULTADOS
    hoje = pd.to_datetime('today').normalize()
    datas_previsao = pd.date_range(start=hoje, periods=len(niveis_previstos))
    df_previsao_final = pd.DataFrame({'data': datas_previsao, 'nivel_m': niveis_previstos})

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