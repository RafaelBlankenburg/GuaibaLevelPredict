# main.py (VERSÃO FINAL SIMPLIFICADA)

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
NUM_DIAS_HISTORICO = NUM_LAGS_MODELO + DIAS_ROLLING_MAX

NUM_DIAS_PREVISAO_CHUVA = 14
DIAS_ADICIONAIS_ESTIMATIVA = 10
DIAS_TOTAIS_PREVISAO = NUM_DIAS_PREVISAO_CHUVA + DIAS_ADICIONAIS_ESTIMATIVA
COTA_INUNDACAO = 3.0
NIVEL_MINIMO_ESTIAGEM = 0.6

def run_prediction_scenarios():
    print("--- INICIANDO ROTINA DE PREVISÃO (LÓGICA SIMPLIFICADA E CORRETA) ---")
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

    df_chuva_total = coletar_dados_chuva(NUM_DIAS_HISTORICO, NUM_DIAS_PREVISAO_CHUVA)
    nivel_atual = coletar_nivel_atual_rio()
    
    print("\n--- Cenário 1: PREVISÃO COM ESTIAGEM (SEM CHUVA APÓS D14) ---")
    df_chuva_cenario1 = df_chuva_total.copy()
    if DIAS_ADICIONAIS_ESTIMATIVA > 0:
        datas_futuras = pd.date_range(start=df_chuva_cenario1.index[-1] + pd.Timedelta(days=1), periods=DIAS_ADICIONAIS_ESTIMATIVA, freq='D')
        df_zeros = pd.DataFrame(0, index=datas_futuras, columns=[col for col in df_chuva_cenario1.columns if col.endswith('_mm')])
        df_chuva_cenario1 = pd.concat([df_chuva_cenario1, df_zeros])

    df_chuva_cenario1['altura_rio_guaiba_m'] = 0 
    df_features_processadas = preprocess_dataframe(df_chuva_cenario1, coluna_nivel='altura_rio_guaiba_m')
    df_features_processadas.set_index('data', inplace=True)

    previsoes = []
    nivel_anterior = nivel_atual
    hoje = pd.to_datetime('today').normalize()
    
    print("🔮 Simulando previsão dia a dia...")
    for data_previsao in pd.date_range(start=hoje, periods=DIAS_TOTAIS_PREVISAO):
        fim_janela = data_previsao - pd.Timedelta(days=1)
        inicio_janela = fim_janela - pd.Timedelta(days=NUM_LAGS_MODELO - 1)
        
        janela_features = df_features_processadas.loc[inicio_janela:fim_janela]
        
        X = janela_features[FEATURES_ENTRADA]
        X_scaled = scaler_entradas.transform(X)
        X_input = np.expand_dims(X_scaled, axis=0)

        delta_scaled = model.predict(X_input, verbose=0)[0][0]
        delta_previsto = scaler_saida.inverse_transform([[delta_scaled]])[0][0]
        
        nivel_previsto = nivel_anterior + delta_previsto
        nivel_previsto = max(NIVEL_MINIMO_ESTIAGEM, nivel_previsto)
        previsoes.append({'data': data_previsao, 'nivel_m': nivel_previsto})
        
        nivel_anterior = nivel_previsto

    df_previsao_final = pd.DataFrame(previsoes)

    gerar_grafico_previsao(df_previsao=df_previsao_final, ponto_de_corte=NUM_DIAS_PREVISAO_CHUVA, cota_inundacao=COTA_INUNDACAO, caminho_saida='results/previsao_nivel_rio.png')

    df_previsao_texto = df_previsao_final.copy()
    df_previsao_texto['nivel_m'] = df_previsao_texto['nivel_m'].round(2)
    df_previsao_texto['data'] = pd.to_datetime(df_previsao_texto['data']).dt.strftime('%d/%m/%Y')
    
    print(f"\n📈 Previsões para {NUM_DIAS_PREVISAO_CHUVA} dias e Estimativas para mais {DIAS_ADICIONAIS_ESTIMATIVA} dias:\n")
    print(df_previsao_texto.to_string(index=False))
    
    df_previsao_texto.to_csv("results/previsao_nivel_rio_com_estimativa.csv", index=False)
    print("\n✅ Previsões salvas em results/previsao_nivel_rio_com_estimativa.csv")

if __name__ == "__main__":
    run_prediction_scenarios()