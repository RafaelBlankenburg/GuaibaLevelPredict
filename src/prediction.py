import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import json # Importa a biblioteca JSON

def prever_nivel_rio_sequencia(chuva_df, nivel_inicial_rio, num_dias_historico, num_dias_previsao, num_lags_modelo):
    """
    Prevê o nível do rio usando um modelo que prevê a VARIAÇÃO (delta).
    Versão robusta que lê os nomes das colunas de um arquivo de configuração.
    """
    print("\n🧠 Carregando modelo DELTA, scalers e configuração de colunas...")
    model = tf.keras.models.load_model('models/lstm_model_delta.keras')
    scaler_entradas = joblib.load('models/scaler_entradas.pkl')
    scaler_delta = joblib.load('models/scaler_delta.pkl')

    # ### ALTERADO: Lendo os nomes das colunas do arquivo JSON ###
    with open('models/training_columns.json', 'r') as f:
        config_colunas = json.load(f)
    
    FEATURES_ENTRADA = config_colunas['features_entrada']
    COLUNA_NIVEL_ABSOLUTO = config_colunas['coluna_nivel_absoluto']
    colunas_chuva = [col for col in FEATURES_ENTRADA if col not in [COLUNA_NIVEL_ABSOLUTO] and 'delta_' not in col and 'acum_' not in col]
    
    historico_chuva = chuva_df.iloc[num_dias_historico - num_lags_modelo : num_dias_historico][colunas_chuva]
    historico_nivel = pd.DataFrame(
        np.full((num_lags_modelo, 1), nivel_inicial_rio),
        index=historico_chuva.index, columns=[COLUNA_NIVEL_ABSOLUTO]
    )
    historico_completo = pd.concat([historico_chuva, historico_nivel], axis=1)

    # Adiciona colunas de features de engenharia com valores iniciais (zero ou calculado)
    for cidade in colunas_chuva:
        historico_completo[f'delta_{cidade}'] = historico_completo[cidade].diff().fillna(0)
        historico_completo[f'acum_{cidade}_3d'] = historico_completo[cidade].rolling(window=3).sum().fillna(0)

    # Garante que as colunas estão na mesma ordem do treinamento
    historico_completo = historico_completo[FEATURES_ENTRADA]

    previsoes_finais_absolutas = []
    ultimo_nivel_conhecido = nivel_inicial_rio
    
    print("🔮 Iniciando previsão autorregressiva com modelo DELTA...")
    
    for i in range(num_dias_previsao):
        janela_padronizada = scaler_entradas.transform(historico_completo)
        janela_lstm_input = np.expand_dims(janela_padronizada, axis=0)
        delta_previsto_scaled = model.predict(janela_lstm_input, verbose=0)[0][0]
        delta_previsto_real = scaler_delta.inverse_transform([[delta_previsto_scaled]])[0][0]
        
        nivel_absoluto_previsto = ultimo_nivel_conhecido + delta_previsto_real
        previsoes_finais_absolutas.append(nivel_absoluto_previsto)
        ultimo_nivel_conhecido = nivel_absoluto_previsto
        
        proximo_passo_chuva = chuva_df.iloc[num_dias_historico + i][colunas_chuva]
        nova_linha_dict = proximo_passo_chuva.to_dict()
        nova_linha_dict[COLUNA_NIVEL_ABSOLUTO] = nivel_absoluto_previsto
        
        nova_linha = pd.DataFrame([nova_linha_dict], index=[chuva_df.index[num_dias_historico + i]])
        historico_temp = pd.concat([historico_completo.iloc[1:], nova_linha])
        
        # Recalcula as features de engenharia para a nova janela
        for cidade in colunas_chuva:
            historico_temp[f'delta_{cidade}'] = historico_temp[cidade].diff().fillna(0)
            historico_temp[f'acum_{cidade}_3d'] = historico_temp[cidade].rolling(window=3).sum().fillna(0)
        
        historico_completo = historico_temp[FEATURES_ENTRADA]

    datas_previsao = chuva_df.index[num_dias_historico : num_dias_historico + num_dias_previsao]
    resultado_df = pd.DataFrame({'data': datas_previsao.strftime('%Y-%m-%d'), 'altura_prevista_m': previsoes_finais_absolutas})
    
    return resultado_df