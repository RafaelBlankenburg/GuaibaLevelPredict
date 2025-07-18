# main.py (Versão Completa e Corrigida)

import pandas as pd
import numpy as np
import os 
import json

# --- Alteração 1: Adicionar importações para carregar o modelo e scalers ---
import joblib
import tensorflow as tf
# --- Fim da Alteração 1 ---

from src.data_collection import coletar_dados_chuva, coletar_nivel_atual_rio
from src.prediction import prever_nivel_rio_sequencia
from src.visualization import gerar_grafico_previsao
from src.preprocess_dataframe import preprocess_dataframe


# --- CONFIGURAÇÕES GERAIS DA EXECUÇÃO ---
NUM_DIAS_HISTORICO = 14
NUM_LAGS_MODELO = 10
NUM_DIAS_PREVISAO_CHUVA = 14
DIAS_ADICIONAIS_ESTIMATIVA = 10
DIAS_TOTAIS_PREVISAO = NUM_DIAS_PREVISAO_CHUVA + DIAS_ADICIONAIS_ESTIMATIVA
COTA_INUNDACAO = 3.0 # Usando 3.0 como definido no seu código de visualização

def run_prediction_scenarios():
    """
    Orquestra a execução da pipeline, baseada em cenários e com separação
    visual entre previsão e estimativa.
    """
    print("--- INICIANDO ROTINA DE PREVISÃO E ESTIMATIVA DO NÍVEL DO RIO ---")

    os.makedirs('results', exist_ok=True) 

    # --- Alteração 2: Carregar o modelo e os scalers salvos pelo train.py ---
    print("\n--- Carregando modelo e scalers salvos ---")
    try:
        model = tf.keras.models.load_model('models/lstm_model_absolute.keras')
        scaler_entradas = joblib.load('models/scaler_entradas.pkl')
        # Atenção: o arquivo de treino salva como 'scaler_alvo.pkl'
        scaler_saida = joblib.load('models/scaler_alvo.pkl') 
        print("✅ Modelo e scalers carregados com sucesso.")
    except IOError as e:
        print(f"❌ Erro ao carregar os arquivos do modelo: {e}")
        print("Por favor, certifique-se de que o script 'train.py' foi executado com sucesso e os arquivos estão na pasta 'models/'.")
        return # Encerra a execução se os arquivos não forem encontrados
    # --- Fim da Alteração 2 ---

    df_chuva_base = coletar_dados_chuva(NUM_DIAS_HISTORICO, NUM_DIAS_PREVISAO_CHUVA)
    nivel_atual = coletar_nivel_atual_rio()

    print("\n---  Cenário 1: PREVISÃO COM ESTIAGEM (SEM CHUVA APÓS D14) ---")
    df_chuva_cenario1 = df_chuva_base.copy()

    if DIAS_ADICIONAIS_ESTIMATIVA > 0:
        datas_futuras = pd.to_datetime(pd.date_range(start=df_chuva_cenario1.index[-1] + pd.Timedelta(days=1), periods=DIAS_ADICIONAIS_ESTIMATIVA, freq='D'))
        df_zeros = pd.DataFrame(0, index=datas_futuras, columns=df_chuva_cenario1.columns)
        df_chuva_cenario1 = pd.concat([df_chuva_cenario1, df_zeros])

    df_chuva_cenario1["altura_rio_guaiba_m"] = nivel_atual
    df_chuva_cenario1 = preprocess_dataframe(df_chuva_cenario1, coluna_nivel='altura_rio_guaiba_m')

    with open("models/training_columns.json", "r") as f:
        colunas_info = json.load(f)

    features_entrada = colunas_info["features_entrada"]

    # Esta chamada agora funcionará, pois as variáveis 'model', 'scaler_entradas' e 'scaler_saida' existem.
    previsao_total_numerica = prever_nivel_rio_sequencia(
        df=df_chuva_cenario1,
        coluna_nivel="altura_rio_guaiba_m",
        FEATURES_ENTRADA=features_entrada,
        scaler_entradas=scaler_entradas,
        scaler_saida=scaler_saida,
        model=model,
        num_lags_modelo=NUM_LAGS_MODELO
    )
    # O nome da coluna previsto na função de predição é 'nivel_previsto', vamos renomear
    previsao_total_numerica.rename(columns={'nivel_previsto': 'nivel_m'}, inplace=True)

    # --- Chamada para gerar o gráfico ---
    gerar_grafico_previsao(
        df_previsao=previsao_total_numerica,
        ponto_de_corte=NUM_DIAS_PREVISAO_CHUVA,
        cota_inundacao=COTA_INUNDACAO,
        caminho_saida='results/previsao_nivel_rio.png'
    )

    # --- Preparação para o print no console ---
    df_previsao_real = previsao_total_numerica.iloc[:NUM_DIAS_PREVISAO_CHUVA]
    df_estimativa = previsao_total_numerica.iloc[NUM_DIAS_PREVISAO_CHUVA:]

    # Cria uma linha de texto para separar as seções
    linha_separadora_data = df_previsao_real['data'].iloc[-1] + pd.Timedelta(days=1)
    linha_separadora = pd.DataFrame([{'data': '--- Início da Estimativa ---', 'nivel_m': '---'}])
    
    # Concatena os dataframes para exibição
    df_resultado_final = pd.concat([df_previsao_real, linha_separadora, df_estimativa], ignore_index=True)

    # Salva o CSV
    df_resultado_final.to_csv("results/previsao_nivel_rio_com_estimativa.csv", index=False)
    print("\n✅ Previsões e estimativas em texto salvas em results/previsao_nivel_rio_com_estimativa.csv")

    print(f"\n📈 Previsões para {NUM_DIAS_PREVISAO_CHUVA} dias e Estimativas para mais {DIAS_ADICIONAIS_ESTIMATIVA} dias:\n")

    # Formata a coluna numérica para exibição, tratando o texto do separador
    df_resultado_final['nivel_m'] = pd.to_numeric(df_resultado_final['nivel_m'], errors='coerce').round(2)
    df_resultado_final['nivel_m'] = df_resultado_final['nivel_m'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else '---')
    df_resultado_final.loc[df_resultado_final['data'] == '--- Início da Estimativa ---', 'nivel_m'] = '---'
    
    # Formata a data
    df_resultado_final['data'] = pd.to_datetime(df_resultado_final['data'], errors='coerce').dt.strftime('%d/%m/%Y')
    df_resultado_final.loc[df_resultado_final['data'].isna(), 'data'] = '--- Início da Estimativa ---'


    print(df_resultado_final.to_string(index=False))


if __name__ == "__main__":
    run_prediction_scenarios()