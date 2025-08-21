# src/train.py (VERSÃO FINAL - TREINO BALANCEADO)

import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os
import json

from preprocess_dataframe import preprocess_dataframe

# --- CONSTANTES ---
NUM_LAGS = 6
RANDOM_SEED = 42
ARQUIVO_CSV = 'data/rain.csv'
FATOR_PESO = 30.0 # Um valor forte e balanceado

def gerar_janelas(df, num_lags, cols_features, col_alvo):
    X, y = [], []
    for i in range(num_lags, len(df)):
        janela_x = df[cols_features].iloc[i - num_lags: i].values
        X.append(janela_x)
        valor_y = df[col_alvo].iloc[i]
        y.append(valor_y)
    return np.array(X), np.array(y)

def train_model():
    print(f"⚙️  Iniciando TREINAMENTO BALANCEADO (Lags={NUM_LAGS}, Fator Peso={FATOR_PESO})...")
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    df_bruto = pd.read_csv(ARQUIVO_CSV)
    df_bruto.dropna(how='all', inplace=True)
    
    COLUNA_NIVEL_ABSOLUTO = 'altura_rio_guaiba_m'
    COLUNA_ALVO_DELTA = 'delta_nivel'
    df_bruto[COLUNA_ALVO_DELTA] = df_bruto[COLUNA_NIVEL_ABSOLUTO].diff()
    
    num_dias = len(df_bruto)
    datas_sinteticas = pd.date_range(start='2020-01-01', periods=num_dias, freq='D')
    df_bruto['data'] = datas_sinteticas

    df = preprocess_dataframe(df_bruto, coluna_nivel=COLUNA_NIVEL_ABSOLUTO)
    FEATURES_ENTRADA = [col for col in df.columns if col not in [COLUNA_NIVEL_ABSOLUTO, COLUNA_ALVO_DELTA, 'data']]

    scaler_entradas = MinMaxScaler()
    scaler_delta = MinMaxScaler()

    entradas_para_scaler = df[FEATURES_ENTRADA]
    alvo_para_scaler = df[[COLUNA_ALVO_DELTA]]
    scaler_entradas.fit(entradas_para_scaler)
    scaler_delta.fit(alvo_para_scaler)

    df_scaled = pd.concat([
        pd.DataFrame(scaler_entradas.transform(entradas_para_scaler), columns=FEATURES_ENTRADA),
        pd.DataFrame(scaler_delta.transform(alvo_para_scaler), columns=[COLUNA_ALVO_DELTA])
    ], axis=1)

    X, y_scaled = gerar_janelas(df_scaled, NUM_LAGS, FEATURES_ENTRADA, COLUNA_ALVO_DELTA)
    _, y_original_delta = gerar_janelas(df, NUM_LAGS, FEATURES_ENTRADA, COLUNA_ALVO_DELTA)
    
    pesos_totais = 1 + np.power(np.abs(y_original_delta) * FATOR_PESO, 2)

    X_train, X_test, y_train, y_test, pesos_train, _ = train_test_split(
        X, y_scaled, pesos_totais, test_size=0.2, random_state=RANDOM_SEED, shuffle=False
    )
    
    print("ℹ️  Aplicando aumento de dados com ruído para robustez...")
    noise_factor = 0.02
    X_train_augmented = X_train + np.random.normal(0, noise_factor, X_train.shape)
    X_train_augmented = np.clip(X_train_augmented, 0, 1)

    num_features = len(FEATURES_ENTRADA)
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(80, return_sequences=True, input_shape=(NUM_LAGS, num_features)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(40),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    
    print(f"🧠 Treinando modelo com dados aumentados...")
    model.fit(
        X_train_augmented, y_train, epochs=200, batch_size=10, validation_data=(X_test, y_test),
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True)],
        verbose=1, sample_weight=pesos_train
    )

    print("💾 Salvando modelo robusto e scalers...")
    os.makedirs('models', exist_ok=True)
    model.save('models/lstm_model_delta.keras')
    joblib.dump(scaler_entradas, 'models/scaler_entradas.pkl')
    joblib.dump(scaler_delta, 'models/scaler_delta.pkl')
    with open('models/training_columns.json', 'w') as f:
        json.dump({'features_entrada': FEATURES_ENTRADA}, f)
    print("✅ Modelo DELTA salvo com sucesso!")

if __name__ == '__main__':
    train_model()