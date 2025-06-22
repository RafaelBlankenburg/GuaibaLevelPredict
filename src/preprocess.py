import numpy as np
import pandas as pd

def gerar_janelas_lstm(df: pd.DataFrame, num_lags: int, coluna_alvo: str, colunas_features: list):
    X, y = [], []

    for i in range(len(df) - num_lags):
        janela_x = df[colunas_features].iloc[i : i + num_lags].values

        valor_y = df[coluna_alvo].iloc[i + num_lags]
        
        X.append(janela_x)
        y.append(valor_y)
        
    return np.array(X), np.array(y)