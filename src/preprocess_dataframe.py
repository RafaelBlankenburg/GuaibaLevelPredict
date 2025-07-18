# src/preprocess_dataframe.py (VERSÃO CORRIGIDA E OTIMIZADA)

import pandas as pd

def preprocess_dataframe(df: pd.DataFrame, coluna_nivel: str, incluir_variaveis=True):
    df_original = df.copy()

    # Se o índice for de datas, move para uma coluna 'data'
    if isinstance(df_original.index, pd.DatetimeIndex):
        df_original = df_original.reset_index().rename(columns={'index': 'data'})

    if not incluir_variaveis:
        return df_original.dropna().reset_index(drop=True)

    # --- CORREÇÃO DO BUG PRINCIPAL ---
    # Identifica as cidades de forma correta e segura
    cidades = [col for col in df_original.columns if str(col).endswith('_mm')]
    
    novas_colunas = {}

    # Cria todas as novas features em um dicionário (mais rápido)
    for cidade in cidades:
        novas_colunas[f'delta_{cidade}'] = df_original[cidade].diff()
        novas_colunas[f'acum_{cidade}_3d'] = df_original[cidade].rolling(window=3).sum()
        novas_colunas[f'acum_{cidade}_5d'] = df_original[cidade].rolling(window=5).sum()
        novas_colunas[f'acum_{cidade}_7d'] = df_original[cidade].rolling(window=7).sum()
        novas_colunas[f'acum_{cidade}_10d'] = df_original[cidade].rolling(window=10).sum()

    if coluna_nivel in df_original.columns:
        novas_colunas['tendencia_5d'] = df_original[coluna_nivel].diff().rolling(window=5).sum()

    df_novas = pd.DataFrame(novas_colunas)

    # Junta o dataframe original com as novas colunas de uma só vez
    df_final = pd.concat([df_original, df_novas], axis=1)

    # Remove linhas com valores NaN e reseta o índice
    df_final = df_final.dropna().reset_index(drop=True)
    
    return df_final