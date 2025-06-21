import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.preprocess import gerar_janelas_lstm

NUM_LAGS = 7
RANDOM_SEED = 42
ARQUIVO_CSV = 'data/rain.csv'

def train_model():
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    # Carregar dados reais
    df = pd.read_csv(ARQUIVO_CSV)

    # Padronizar os afluentes
    afluentes = [col for col in df.columns if col.startswith('afluente_')]
    scaler = StandardScaler()
    df[afluentes] = scaler.fit_transform(df[afluentes])

    X, y = gerar_janelas_lstm(df, NUM_LAGS)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )

    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(NUM_LAGS, len(afluentes))),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=30, batch_size=16, validation_split=0.2)

    print(f'Erro de teste: {model.evaluate(X_test, y_test):.2f}')

    model.save('models/lstm_model.keras')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(afluentes, 'models/afluentes.pkl')
