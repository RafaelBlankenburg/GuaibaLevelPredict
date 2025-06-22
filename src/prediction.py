# Arquivo: src/prediction.py (modificado)

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

def prever_nivel_rio_sequencia(chuva_df, nivel_inicial_rio, num_dias_historico, num_dias_previsao, num_lags_modelo):
    """
    Prevê o nível do rio de forma autorregressiva, usando o histórico 
    completo de chuva E NÍVEL como entrada.
    """
    print("\n🧠 Carregando modelo e scalers (versão com histórico completo)...")
    model = tf.keras.models.load_model('models/lstm_model.keras')
    scaler_chuva = joblib.load('models/scaler_chuva.pkl')
    scaler_nivel = joblib.load('models/scaler_nivel.pkl')

    colunas_chuva = scaler_chuva.feature_names_in_.tolist()
    coluna_alvo = scaler_nivel.feature_names_in_[0]

    # ### ALTERADO: Lógica de construção do histórico inicial ###
    # Precisamos de um histórico inicial para o nível do rio. 
    # Como não temos o histórico real, vamos assumir que o rio estava estável
    # no nível inicial nos últimos N dias. Esta é uma premissa que podemos refinar depois.
    historico_chuva = chuva_df.iloc[num_dias_historico - num_lags_modelo : num_dias_historico]
    historico_nivel = pd.DataFrame(
        np.full((num_lags_modelo, 1), nivel_inicial_rio),
        index=historico_chuva.index,
        columns=[coluna_alvo]
    )
    
    # Este DataFrame conterá a janela de dados que deslizará para o futuro
    historico_completo = pd.concat([historico_chuva, historico_nivel], axis=1)

    previsoes_finais = []
    print("🔮 Iniciando previsão autorregressiva com histórico completo...")
    
    for i in range(num_dias_previsao):
        # Passo 1: Padronizar a janela de entrada atual
        chuva_scaled = scaler_chuva.transform(historico_completo[colunas_chuva])
        nivel_scaled = scaler_nivel.transform(historico_completo[[coluna_alvo]])
        
        janela_scaled = np.concatenate([chuva_scaled, nivel_scaled], axis=1)
        
        # Passo 2: Fazer a previsão
        janela_lstm_input = np.expand_dims(janela_scaled, axis=0)
        pred_scaled = model.predict(janela_lstm_input, verbose=0)[0][0]
        
        # Passo 3: Despadronizar a previsão
        pred_descaled = scaler_nivel.inverse_transform([[pred_scaled]])[0][0]
        previsoes_finais.append(pred_descaled)
        
        # Passo 4: ATUALIZAR O HISTÓRICO PARA A PRÓXIMA PREVISÃO
        # Pegamos os dados de chuva do próximo dia
        proximo_passo_chuva = chuva_df.iloc[num_dias_historico + i]
        
        # Criamos uma nova linha com a chuva prevista e o nível previsto
        nova_linha = pd.DataFrame([proximo_passo_chuva.values.tolist() + [pred_descaled]],
                                  columns=colunas_chuva + [coluna_alvo],
                                  index=[proximo_passo_chuva.name])

        # Adicionamos a nova linha e removemos a mais antiga para deslizar a janela
        historico_completo = pd.concat([historico_completo, nova_linha])
        historico_completo = historico_completo.iloc[1:]

    datas_previsao = chuva_df.index[num_dias_historico : num_dias_historico + num_dias_previsao]
    resultado_df = pd.DataFrame({'data': datas_previsao.strftime('%Y-%m-%d'), 'altura_prevista_m': previsoes_finais})
    
    return resultado_df