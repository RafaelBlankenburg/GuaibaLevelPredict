# Arquivo: src/prediction.py

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from src.data_collection import CIDADES # Importa a lista de cidades

def prever_nivel_rio_sequencia(chuva_df, nivel_inicial_rio, num_dias_historico, num_dias_previsao, num_lags_modelo):
    """
    Prevê o nível do rio de forma autorregressiva, construindo cada janela de
    entrada manualmente para espelhar a lógica de treinamento.
    """
    print("\n🧠  Carregando modelo e scaler treinados...")
    model = tf.keras.models.load_model('models/lstm_model.keras')
    scaler = joblib.load('models/scaler.pkl')
    
    colunas_chuva = [c[0] for c in CIDADES]
    coluna_alvo = 'altura_rio_guaiba_m'
    
    # Inicia o histórico de previsões com o último nível conhecido
    ultimo_nivel_conhecido = nivel_inicial_rio
    previsoes_finais = []
    
    print("🔮  Iniciando previsão autorregressiva para os próximos dias...")
    for i in range(num_dias_previsao):
        # Passo 1: Preparar a janela de features de chuva
        # Pega os últimos 'num_lags_modelo' dias de chuva disponíveis
        offset_inicio = num_dias_historico + i - num_lags_modelo
        offset_fim = num_dias_historico + i
        janela_chuva = chuva_df.iloc[offset_inicio:offset_fim][colunas_chuva].values
        
        # Passo 2: Preparar a feature de nível do rio
        # O modelo espera o nível do rio do dia anterior como uma feature constante em toda a janela.
        nivel_rio_feature = np.full((num_lags_modelo, 1), ultimo_nivel_conhecido)
        
        # Passo 3: Combinar as features para espelhar os dados de treino
        janela_combinada = np.concatenate((janela_chuva, nivel_rio_feature), axis=1)

        # Passo 4: Padronizar os dados da janela
        # O scaler espera um DataFrame com os nomes de coluna corretos
        df_para_scaler = pd.DataFrame(janela_combinada, columns=colunas_chuva + [coluna_alvo])
        janela_scaled = scaler.transform(df_para_scaler)
        
        # Passo 5: Fazer a previsão
        # O shape precisa ser (1, num_lags, num_features) para o LSTM
        janela_lstm_input = np.expand_dims(janela_scaled, axis=0)
        pred_scaled = model.predict(janela_lstm_input, verbose=0)[0][0]
        
        # Passo 6: "Despadronizar" a previsão para obter o valor real em metros
        # Criamos um array "dummy" com o shape esperado pelo scaler para a transformação inversa
        dummy_array = np.zeros((1, len(colunas_chuva) + 1))
        dummy_array[0, -1] = pred_scaled  # Coloca a previsão na última posição (do alvo)
        pred_descaled = scaler.inverse_transform(dummy_array)[0][-1]
        
        # Passo 7: Guardar a previsão e atualizar o 'ultimo_nivel_conhecido' para a próxima iteração
        previsoes_finais.append(pred_descaled)
        ultimo_nivel_conhecido = pred_descaled

    # Criar o dataframe final de resultados
    datas_previsao = chuva_df.index[num_dias_historico : num_dias_historico + num_dias_previsao]
    resultado_df = pd.DataFrame({'data': datas_previsao.strftime('%Y-%m-%d'), 'altura_prevista_m': previsoes_finais})
    
    return resultado_df