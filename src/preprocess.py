import numpy as np
import pandas as pd

def gerar_janelas_lstm(df: pd.DataFrame, num_lags: int):
    """
    Gera X com formato (amostras, passos de tempo, afluentes) e y com altura do rio
    """
    afluentes = [col for col in df.columns if col.startswith('afluente_')]
    X, y = [], []

    for i in range(num_lags, len(df)):
        janela = df[afluentes].iloc[i - num_lags:i].values
        X.append(janela)
        y.append(df['altura_rio'].iloc[i])

    return np.array(X), np.array(y)
