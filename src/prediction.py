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

    df = df.copy()
    df = df.dropna().reset_index(drop=True)

    num_dias_historico = len(df)
    dias_previstos = len(df) - num_lags_modelo

    for i in range(dias_previstos):
        historico_completo = df.iloc[i : i + num_lags_modelo + 1].copy()
        historico_completo = historico_completo.dropna().reset_index(drop=True)

        if len(historico_completo) < num_lags_modelo + 1:
            continue

        historico_completo_scaled = scaler_entradas.transform(historico_completo[FEATURES_ENTRADA])
        X_input = historico_completo_scaled[:-1]  # Último dia é o alvo
        X_input = np.expand_dims(X_input, axis=0)

        y_pred_scaled = model.predict(X_input, verbose=0)[0][0]
        y_pred = scaler_saida.inverse_transform([[y_pred_scaled]])[0][0]

        data_prevista = historico_completo.iloc[-1].name
        df_resultado.append((data_prevista, y_pred))

    df_previsto = pd.DataFrame(df_resultado, columns=['data', 'nivel_previsto'])
    return df_previsto
