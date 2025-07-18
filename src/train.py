# Arquivo: src/train.py (Versão Corrigida)

import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import json
from preprocess_dataframe import preprocess_dataframe

# --- CONSTANTES ---
NUM_LAGS = 14
RANDOM_SEED = 42
ARQUIVO_CSV = 'data/rain.csv'
FATOR_PESO = 3.0


def gerar_janelas(df, num_lags, cols_features, col_alvo):
    X, y = [], []
    for i in range(num_lags, len(df)):
        janela_x = df[cols_features].iloc[i - num_lags: i].values
        X.append(janela_x)
        valor_y = df[col_alvo].iloc[i]
        y.append(valor_y)
    return np.array(X), np.array(y)


def train_model():
    print("⚙️  Iniciando processo de treinamento com previsão absoluta...")
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    df = pd.read_csv(ARQUIVO_CSV)
    df.dropna(how='all', inplace=True)

    COLUNA_NIVEL_ABSOLUTO = 'altura_rio_guaiba_m'
    FEATURES_ENTRADA = [col for col in df.columns if col != COLUNA_NIVEL_ABSOLUTO and col != 'data']

    scaler_entradas = StandardScaler()
    scaler_alvo = StandardScaler()

    entradas_para_scaler = df[FEATURES_ENTRADA]
    alvo_para_scaler = df[[COLUNA_NIVEL_ABSOLUTO]]
    scaler_entradas.fit(entradas_para_scaler)
    scaler_alvo.fit(alvo_para_scaler)

    df_scaled_entradas = pd.DataFrame(scaler_entradas.transform(entradas_para_scaler), columns=FEATURES_ENTRADA)
    df_scaled_alvo = pd.DataFrame(scaler_alvo.transform(alvo_para_scaler), columns=[COLUNA_NIVEL_ABSOLUTO])
    df_scaled = pd.concat([df_scaled_entradas, df_scaled_alvo], axis=1)

    X, y_scaled = gerar_janelas(df_scaled, NUM_LAGS, FEATURES_ENTRADA, COLUNA_NIVEL_ABSOLUTO)
    _, y_original = gerar_janelas(df, NUM_LAGS, FEATURES_ENTRADA, COLUNA_NIVEL_ABSOLUTO)

    pesos_totais = 1 + np.abs(pd.Series(y_original).diff().fillna(0)) * FATOR_PESO

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

    model.compile(optimizer='adam', loss='mse')

    print(f"🧠 Treinando com {len(X_train)} amostras e {num_features} features...")
    model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=16,
        validation_data=(X_test, y_test),
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
        verbose=1,
        sample_weight=pesos_train
    )

    print("💾 Salvando modelo e escalers...")
    os.makedirs('models', exist_ok=True)
    model.save('models/lstm_model_absolute.keras')
    joblib.dump(scaler_entradas, 'models/scaler_entradas.pkl')
    joblib.dump(scaler_alvo, 'models/scaler_alvo.pkl')

    with open('models/training_columns.json', 'w') as f:
        json.dump({
            'features_entrada': FEATURES_ENTRADA,
            'coluna_nivel_absoluto': COLUNA_NIVEL_ABSOLUTO
        }, f)

    print("✅ Modelo salvo com sucesso!")


if __name__ == '__main__':
    train_model()
