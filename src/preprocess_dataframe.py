# src/preprocess_dataframe.py

import pandas as pd

def preprocess_dataframe(df: pd.DataFrame, coluna_nivel: str, incluir_variaveis=True):
    df = df.copy()

    cidades = [col for col in df.columns if col != coluna_nivel]

    if incluir_variaveis:
        for cidade in cidades:
            df[f'delta_{cidade}'] = df[cidade].diff()
            df[f'acum_{cidade}_3d'] = df[cidade].rolling(window=3).sum()
            df[f'acum_{cidade}_5d'] = df[cidade].rolling(window=5).sum()
            df[f'acum_{cidade}_7d'] = df[cidade].rolling(window=7).sum()
            df[f'acum_{cidade}_10d'] = df[cidade].rolling(window=10).sum()

        if coluna_nivel in df.columns:
            df['tendencia_5d'] = df[coluna_nivel].diff().rolling(window=5).sum()

    df = df.dropna().reset_index(drop=True)
    return df
