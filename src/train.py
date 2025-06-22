# Arquivo: src/train.py (modificado para prever DELTA)

import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

NUM_LAGS = 14
RANDOM_SEED = 60
ARQUIVO_CSV = 'data/rain.csv'

def gerar_janelas(df, num_lags, cols_features, col_alvo_delta):
    X, y = [], []
    for i in range(num_lags, len(df)):
        janela_x = df[cols_features].iloc[i - num_lags : i].values
        X.append(janela_x)
        valor_y = df[col_alvo_delta].iloc[i]
        y.append(valor_y)
    return np.array(X), np.array(y)

def train_model():
    print("⚙️  Iniciando treinamento para prever a VARIAÇÃO (DELTA) do nível...")
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    df = pd.read_csv(ARQUIVO_CSV)
    df.dropna(inplace=True) # Garante que não há NaNs

    COLUNA_NIVEL_ABSOLUTO = 'altura_rio_guaiba_m'
    
    # ### NOVO: CRIANDO A COLUNA ALVO 'DELTA' ###
    # Nosso novo alvo é a diferença do nível de um dia para o outro.
    COLUNA_ALVO_DELTA = 'delta_nivel'
    df[COLUNA_ALVO_DELTA] = df[COLUNA_NIVEL_ABSOLUTO].diff()
    
    # O primeiro valor do delta será NaN, então removemos essa linha.
    df = df.dropna()

    features_chuva = [col for col in df.columns if col not in [COLUNA_NIVEL_ABSOLUTO, COLUNA_ALVO_DELTA]]
    
    # As features de entrada continuam sendo a chuva e o NÍVEL ABSOLUTO do rio
    FEATURES_ENTRADA = features_chuva + [COLUNA_NIVEL_ABSOLUTO]
    
    # Scalers: um para as features de entrada, e um NOVO para o nosso alvo (delta)
    scaler_entradas = StandardScaler()
    scaler_delta = StandardScaler()

    # Treinamos os scalers
    df_scaled_entradas = pd.DataFrame(scaler_entradas.fit_transform(df[FEATURES_ENTRADA]), columns=FEATURES_ENTRADA, index=df.index)
    df_scaled_delta = pd.DataFrame(scaler_delta.fit_transform(df[[COLUNA_ALVO_DELTA]]), columns=[COLUNA_ALVO_DELTA], index=df.index)

    # Juntamos tudo num dataframe padronizado para gerar as janelas
    df_scaled = pd.concat([df_scaled_entradas, df_scaled_delta], axis=1)

    # Geramos as janelas. O alvo agora é a coluna de DELTA.
    X, y = gerar_janelas(df_scaled, NUM_LAGS, FEATURES_ENTRADA, COLUNA_ALVO_DELTA)
    
    print(f"✅ Janelas de dados geradas. Shape de X: {X.shape}, Shape de y: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, shuffle=False
    )

    num_features = len(FEATURES_ENTRADA)
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(NUM_LAGS, num_features)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1) # A saída é um único valor: o delta previsto
    ])

    model.compile(optimizer='adam', loss='mse')
    
    print("🧠  Iniciando o treinamento do modelo LSTM para prever o DELTA...")
    model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test),
              callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)], verbose=1)

    print("💾  Salvando o modelo e os novos artefatos (scaler_entradas, scaler_delta)...")
    model.save('models/lstm_model_delta.keras')
    joblib.dump(scaler_entradas, 'models/scaler_entradas.pkl')
    joblib.dump(scaler_delta, 'models/scaler_delta.pkl')
    print("✅ Modelo e artefatos salvos com sucesso!")

if __name__ == '__main__':
    train_model()