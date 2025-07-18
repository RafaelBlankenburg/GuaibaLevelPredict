# prediction.py (Versão Corrigida)

import numpy as np
import pandas as pd
from typing import List
from sklearn.preprocessing import StandardScaler
import tensorflow as tf


def prever_nivel_rio_sequencia(
    df: pd.DataFrame,
    coluna_nivel: str,
    FEATURES_ENTRADA: List[str],
    scaler_entradas: StandardScaler,
    scaler_saida: StandardScaler,
    model: tf.keras.Model,
    num_lags_modelo: int = 14
):
    df_resultado = []
    
    # Não precisamos mais fazer df.dropna().reset_index() aqui,
    # pois o preprocess_dataframe já fez isso.
    # O df que chega já está limpo e com a coluna 'data'.

    # O número de previsões que podemos fazer é o tamanho do dataframe menos o tamanho da janela do modelo.
    dias_previstos = len(df) - num_lags_modelo

    for i in range(dias_previstos):
        # A janela de dados de entrada para o modelo
        janela_de_entrada = df.iloc[i : i + num_lags_modelo]
        
        # A linha correspondente ao dia que queremos prever
        linha_alvo = df.iloc[i + num_lags_modelo]

        # Pega a data real da coluna 'data'
        data_prevista = linha_alvo['data']

        # Prepara a janela de entrada para o modelo
        X_input = janela_de_entrada[FEATURES_ENTRADA]
        X_input_scaled = scaler_entradas.transform(X_input)
        X_input_scaled = np.expand_dims(X_input_scaled, axis=0)

        # Faz a predição
        y_pred_scaled = model.predict(X_input_scaled, verbose=0)[0][0]
        y_pred = scaler_saida.inverse_transform([[y_pred_scaled]])[0][0]

        df_resultado.append((data_prevista, y_pred))

    df_previsto = pd.DataFrame(df_resultado, columns=['data', 'nivel_previsto'])
    return df_previsto