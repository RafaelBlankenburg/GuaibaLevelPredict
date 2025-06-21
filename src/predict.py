import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from typing import List

NUM_LAGS = 7

def prever_altura(historico_afluentes: List[List[float]]) -> float:
    """
    Recebe uma lista com os últimos 7 dias de medições dos 8 afluentes.
    """
    afluentes = joblib.load('models/afluentes.pkl')
    scaler = joblib.load('models/scaler.pkl')
    model = tf.keras.models.load_model('models/lstm_model.keras')

    assert len(historico_afluentes) == NUM_LAGS, f"Esperado {NUM_LAGS} dias"
    assert all(len(linha) == len(afluentes) for linha in historico_afluentes), \
        f"Esperado {len(afluentes)} afluentes por dia"

    dados = pd.DataFrame(historico_afluentes, columns=afluentes)
    dados_scaled = scaler.transform(dados)
    entrada = dados_scaled.reshape(1, NUM_LAGS, len(afluentes))

    pred = model.predict(entrada)
    return pred[0][0]
