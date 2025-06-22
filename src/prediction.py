# Arquivo: src/prediction.py (modificado para usar o modelo de DELTA)

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

def prever_nivel_rio_sequencia(chuva_df, nivel_inicial_rio, num_dias_historico, num_dias_previsao, num_lags_modelo):
    """
    Prevê o nível do rio usando um modelo que prevê a VARIAÇÃO (delta).
    """
    print("\n🧠 Carregando modelo DELTA e scalers...")
    model = tf.keras.models.load_model('models/lstm_model_delta.keras')
    scaler_entradas = joblib.load('models/scaler_entradas.pkl')
    scaler_delta = joblib.load('models/scaler_delta.pkl')

    # Extrai os nomes das colunas dos scalers
    coluna_nivel_absoluto = 'altura_rio_guaiba_m'
    colunas_chuva = [col for col in scaler_entradas.feature_names_in_ if col != coluna_nivel_absoluto]

    # Cria o histórico inicial para a primeira previsão
    historico_chuva = chuva_df.iloc[num_dias_historico - num_lags_modelo : num_dias_historico][colunas_chuva]
    historico_nivel = pd.DataFrame(
        np.full((num_lags_modelo, 1), nivel_inicial_rio),
        index=historico_chuva.index, columns=[coluna_nivel_absoluto]
    )
    historico_completo = pd.concat([historico_chuva, historico_nivel], axis=1)

    # Variáveis para o loop
    previsoes_finais_absolutas = []
    ultimo_nivel_conhecido = nivel_inicial_rio
    
    print("🔮 Iniciando previsão autorregressiva com modelo DELTA...")
    
    for i in range(num_dias_previsao):
        # Passo 1: Padronizar a janela de entrada atual
        janela_padronizada = scaler_entradas.transform(historico_completo)
        
        # Passo 2: Fazer a previsão do DELTA
        janela_lstm_input = np.expand_dims(janela_padronizada, axis=0)
        delta_previsto_scaled = model.predict(janela_lstm_input, verbose=0)[0][0]
        
        # Passo 3: Despadronizar o DELTA para obter o valor real da variação
        delta_previsto_real = scaler_delta.inverse_transform([[delta_previsto_scaled]])[0][0]
        
        # ### LÓGICA FUNDAMENTAL ###
        # O nível de amanhã é o nível de hoje + a variação prevista.
        nivel_absoluto_previsto = ultimo_nivel_conhecido + delta_previsto_real
        
        # Guarda o resultado final absoluto
        previsoes_finais_absolutas.append(nivel_absoluto_previsto)
        
        # Atualiza o último nível conhecido para a próxima iteração
        ultimo_nivel_conhecido = nivel_absoluto_previsto
        
        # Passo 4: Atualizar a janela de histórico para a próxima previsão
        proximo_passo_chuva = chuva_df.iloc[num_dias_historico + i][colunas_chuva]
        nova_linha = pd.DataFrame([proximo_passo_chuva.tolist() + [nivel_absoluto_previsto]],
                                  columns=colunas_chuva + [coluna_nivel_absoluto],
                                  index=[chuva_df.index[num_dias_historico + i]])

        historico_completo = pd.concat([historico_completo.iloc[1:], nova_linha])

    datas_previsao = chuva_df.index[num_dias_historico : num_dias_historico + num_dias_previsao]
    resultado_df = pd.DataFrame({'data': datas_previsao.strftime('%Y-%m-%d'), 'altura_prevista_m': previsoes_finais_absolutas})
    
    return resultado_df