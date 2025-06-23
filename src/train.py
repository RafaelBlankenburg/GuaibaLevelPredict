# Arquivo: src/train.py

import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import json

# --- CONSTANTES ---
NUM_LAGS = 10
RANDOM_SEED = 42
ARQUIVO_CSV = 'data/rain.csv' 
FATOR_PESO = 4.0

def gerar_janelas(df, num_lags, cols_features, col_alvo_delta):
    X, y = [], []
    for i in range(num_lags, len(df)):
        janela_x = df[cols_features].iloc[i - num_lags : i].values
        X.append(janela_x)
        valor_y = df[col_alvo_delta].iloc[i]
        y.append(valor_y)
    return np.array(X), np.array(y)

def train_model():
    print("⚙️  Iniciando processo de treinamento (versão com correção de features)...")
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    df = pd.read_csv(ARQUIVO_CSV)
    df.dropna(how='all', inplace=True)

    COLUNA_NIVEL_ABSOLUTO = 'altura_rio_guaiba_m'
    features_chuva = [col for col in df.columns if col != COLUNA_NIVEL_ABSOLUTO]

    for cidade in features_chuva:
        df[f'delta_{cidade}'] = df[cidade].diff()
        df[f'acum_{cidade}_3d'] = df[cidade].rolling(window=3).sum()

    COLUNA_ALVO_DELTA = 'delta_nivel'
    df[COLUNA_ALVO_DELTA] = df[COLUNA_NIVEL_ABSOLUTO].diff()
    df = df.dropna().reset_index(drop=True)

    # --- ### ALTERADO: Lógica de construção da lista de features corrigida ### ---
    # Constrói a lista de features de forma explícita para evitar erros.
    engineered_delta_features = [f'delta_{cidade}' for cidade in features_chuva]
    engineered_acum_features = [f'acum_{cidade}_3d' for cidade in features_chuva]
    
    FEATURES_ENTRADA = (
        features_chuva + 
        engineered_delta_features + 
        engineered_acum_features + 
        [COLUNA_NIVEL_ABSOLUTO]
    )
    # Comprimento esperado = 20 (chuva) + 20 (delta) + 20 (acum) + 1 (nível) = 61
    # --- FIM DA ALTERAÇÃO ---

    scaler_entradas = StandardScaler()
    scaler_delta = StandardScaler()

    entradas_para_scaler = df[FEATURES_ENTRADA]
    delta_para_scaler = df[[COLUNA_ALVO_DELTA]]
    scaler_entradas.fit(entradas_para_scaler)
    scaler_delta.fit(delta_para_scaler)
    
    df_scaled_entradas = pd.DataFrame(scaler_entradas.transform(entradas_para_scaler), columns=FEATURES_ENTRADA)
    df_scaled_delta = pd.DataFrame(scaler_delta.transform(delta_para_scaler), columns=[COLUNA_ALVO_DELTA])
    df_scaled = pd.concat([df_scaled_entradas, df_scaled_delta], axis=1)

    X, y_scaled = gerar_janelas(df_scaled, NUM_LAGS, FEATURES_ENTRADA, COLUNA_ALVO_DELTA)
    _, y_original = gerar_janelas(df, NUM_LAGS, FEATURES_ENTRADA, COLUNA_ALVO_DELTA)
    
    if num_positivos := np.sum(y_original > 0):
        fator_base = (len(y_original) - num_positivos) / num_positivos
        fator_final = fator_base * FATOR_PESO
        pesos_totais = np.where(y_original > 0, fator_final, 1)
    else:
        pesos_totais = np.ones(len(y_original))

    X_train, X_test, y_train, y_test, pesos_train, _ = train_test_split(
        X, y_scaled, pesos_totais, test_size=0.2, random_state=RANDOM_SEED, shuffle=False
    )
    
    num_features = len(FEATURES_ENTRADA)
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(NUM_LAGS, num_features)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mae')
    
    print(f"🧠 Iniciando o treinamento com {len(X_train)} amostras e {num_features} features de entrada...")
    model.fit(
        X_train, y_train, 
        epochs=50, 
        batch_size=16, 
        validation_data=(X_test, y_test),
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
        verbose=1,
        sample_weight=pesos_train
    )

    print("💾  Salvando os artefatos...")
    os.makedirs('models', exist_ok=True)
    model.save('models/lstm_model_delta.keras')
    joblib.dump(scaler_entradas, 'models/scaler_entradas.pkl')
    joblib.dump(scaler_delta, 'models/scaler_delta.pkl')

    config_colunas = {
        'features_entrada': FEATURES_ENTRADA,
        'coluna_nivel_absoluto': COLUNA_NIVEL_ABSOLUTO
    }
    with open('models/training_columns.json', 'w') as f:
        json.dump(config_colunas, f)

    print("✅ Modelo e artefatos salvos com sucesso!")


if __name__ == '__main__':
    train_model()