import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

def prever_nivel_rio_sequencia(chuva_df, nivel_inicial_rio, num_dias_historico, num_dias_previsao, num_lags_modelo):
    print("\n🧠 Carregando modelo DELTA e scalers...")
    model = tf.keras.models.load_model('models/lstm_model_delta.keras')
    scaler_entradas = joblib.load('models/scaler_entradas.pkl')
    scaler_delta = joblib.load('models/scaler_delta.pkl')

    coluna_nivel_absoluto = 'altura_rio_guaiba_m'
    colunas_chuva = [col for col in scaler_entradas.feature_names_in_ if col != coluna_nivel_absoluto]


    historico_chuva = chuva_df.iloc[num_dias_historico - num_lags_modelo : num_dias_historico][colunas_chuva]
    historico_nivel = pd.DataFrame(
        np.full((num_lags_modelo, 1), nivel_inicial_rio),
        index=historico_chuva.index, columns=[coluna_nivel_absoluto]
    )
    historico_completo = pd.concat([historico_chuva, historico_nivel], axis=1)


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
        nova_linha = pd.DataFrame([proximo_passo_chuva.tolist() + [nivel_absoluto_previsto]],
                                  columns=colunas_chuva + [coluna_nivel_absoluto],
                                  index=[chuva_df.index[num_dias_historico + i]])

        historico_completo = pd.concat([historico_completo.iloc[1:], nova_linha])

    datas_previsao = chuva_df.index[num_dias_historico : num_dias_historico + num_dias_previsao]
    resultado_df = pd.DataFrame({'data': datas_previsao.strftime('%Y-%m-%d'), 'altura_prevista_m': previsoes_finais_absolutas})
    
    return resultado_df