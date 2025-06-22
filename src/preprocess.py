# Arquivo: src/preprocess.py

import numpy as np
import pandas as pd

def gerar_janelas_lstm(df: pd.DataFrame, num_lags: int, coluna_alvo: str, colunas_features: list):
    """
    Gera janelas de dados (features e target) para um modelo LSTM.

    Args:
        df (pd.DataFrame): O DataFrame contendo os dados já escalados.
        num_lags (int): O número de passos no tempo para usar como entrada (features).
        coluna_alvo (str): O nome da coluna a ser usada como o alvo (y).
        colunas_features (list): Uma lista com os nomes das colunas a serem usadas como features (X).

    Returns:
        tuple[np.ndarray, np.ndarray]: Uma tupla contendo os arrays X (features) e y (target).
    """
    X, y = [], []
    
    # Itera sobre o dataframe, parando antes do final para garantir que haja dados para o alvo
    for i in range(len(df) - num_lags):
        # Pega uma janela de 'num_lags' passos de tempo das colunas de features
        janela_x = df[colunas_features].iloc[i : i + num_lags].values

        valor_y = df[coluna_alvo].iloc[i + num_lags]
        
        X.append(janela_x)
        y.append(valor_y)
        
    return np.array(X), np.array(y)