# src/preprocess_dataframe.py (Versão Corrigida)

import pandas as pd

def preprocess_dataframe(df: pd.DataFrame, coluna_nivel: str, incluir_variaveis=True):
    # Usamos o índice (que contém as datas) para criar uma coluna 'data'
    # Fazemos isso antes de qualquer outra operação para garantir que não se perca
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index().rename(columns={'index': 'data'})

    cidades = [col for col in df.columns if col.endswith('_mm')]

    if incluir_variaveis:
        for cidade in cidades:
            df[f'delta_{cidade}'] = df[cidade].diff()
            df[f'acum_{cidade}_3d'] = df[cidade].rolling(window=3).sum()
            df[f'acum_{cidade}_5d'] = df[cidade].rolling(window=5).sum()
            df[f'acum_{cidade}_7d'] = df[cidade].rolling(window=7).sum()
            df[f'acum_{cidade}_10d'] = df[cidade].rolling(window=10).sum()

        if coluna_nivel in df.columns:
            df['tendencia_5d'] = df[coluna_nivel].diff().rolling(window=5).sum()
    
    # Remove as linhas com NaN geradas pelo rolling window e reseta o índice
    # Desta vez, drop=True não é problema, pois a data já está salva na coluna 'data'
    df = df.dropna().reset_index(drop=True)
    return df