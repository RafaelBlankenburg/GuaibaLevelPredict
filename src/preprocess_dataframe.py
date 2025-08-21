import pandas as pd
import numpy as np

BACIA_TAQUARI_ANTAS = ["vacaria_mm", "sao_francisco_de_paula_mm", "caxias_do_sul_mm", "bento_goncalves_mm", "guapore_mm", "lagoa_vermelha_mm"]
BACIA_JACUI = ["passo_fundo_mm", "soledade_mm", "cruz_alta_mm", "salto_do_jacui_mm", "julio_de_castilhos_mm", "santa_maria_mm", "cachoeira_do_sul_mm"]

def preprocess_dataframe(df: pd.DataFrame, coluna_nivel: str):
    df_original = df.copy()
    if isinstance(df_original.index, pd.DatetimeIndex):
        df_original = df_original.reset_index().rename(columns={'index': 'data'})
    
    has_date_column = 'data' in df_original.columns
    if has_date_column:
        df_original['data'] = pd.to_datetime(df_original['data'])

    cidades = [col for col in df_original.columns if str(col).endswith('_mm')]
    df_features = pd.DataFrame(index=df_original.index)
    
    for cidade in cidades:
        df_features[f'delta_{cidade}'] = df_original[cidade].diff()
        for window in [3, 5, 7]:
            df_features[f'acum_{cidade}_{window}d'] = df_original[cidade].rolling(window=window).sum()

    cidades_taquari_existentes = [c for c in BACIA_TAQUARI_ANTAS if c in df_original.columns]
    df_features['bomba_chuva_taquari_3d'] = df_original[cidades_taquari_existentes].rolling(window=3).sum().sum(axis=1)
    
    cidades_jacui_existentes = [c for c in BACIA_JACUI if c in df_original.columns]
    df_features['bomba_chuva_jacui_3d'] = df_original[cidades_jacui_existentes].rolling(window=3).sum().sum(axis=1)

    if coluna_nivel in df_original.columns:
        df_features['nivel_lag_1'] = df_original[coluna_nivel].shift(1)
        df_features['tendencia_3d'] = df_original[coluna_nivel].diff().rolling(window=3).mean()

    if has_date_column:
        df_features['mes_sin'] = np.sin(2 * np.pi * df_original['data'].dt.month / 12)
        df_features['mes_cos'] = np.cos(2 * np.pi * df_original['data'].dt.month / 12)

    df_final = pd.concat([df_original, df_features], axis=1)
    df_final = df_final.dropna().reset_index(drop=True)
    return df_final