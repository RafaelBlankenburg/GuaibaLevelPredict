# Arquivo: src/train.py

import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- CONSTANTES DO MODELO ---
NUM_LAGS = 14  # Podemos usar 10 dias de lag, como você sugeriu
RANDOM_SEED = 42
ARQUIVO_CSV = 'data/rain.csv'

def gerar_janelas_com_nivel_anterior(df, num_lags, col_alvo, cols_features):
    """
    Gera janelas para o modelo, incluindo o nível do rio do dia anterior como uma feature.
    """
    X, y = [], []
    # O loop começa de 'num_lags' para garantir que temos dados históricos suficientes
    for i in range(num_lags, len(df)):
        # Pega a janela de features de chuva
        janela_features_chuva = df[cols_features].iloc[i - num_lags : i].values
        
        # Pega o nível do rio do dia anterior (i-1)
        nivel_rio_anterior = df[col_alvo].iloc[i - 1]
        
        # Cria uma matriz para o nível do rio com o mesmo shape de lag da chuva
        # e preenche com o valor do dia anterior
        nivel_rio_feature = np.full((num_lags, 1), nivel_rio_anterior)
        
        # Concatena as features de chuva com a nova feature de nível do rio
        janela_final_x = np.concatenate((janela_features_chuva, nivel_rio_feature), axis=1)
        
        X.append(janela_final_x)
        y.append(df[col_alvo].iloc[i])
        
    return np.array(X), np.array(y)


def train_model():
    print("⚙️  Iniciando treinamento do modelo com a nova lógica...")
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    df = pd.read_csv(ARQUIVO_CSV)

    COLUNA_ALVO = 'altura_rio_guaiba_m'
    features_chuva = [col for col in df.columns if col != COLUNA_ALVO]
    
    # IMPORTANTE: O scaler agora deve ser treinado com TODAS as colunas, incluindo o alvo,
    # pois o nível do rio também será uma feature.
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    # Gera as janelas com a nova função
    X, y = gerar_janelas_com_nivel_anterior(df_scaled, NUM_LAGS, COLUNA_ALVO, features_chuva)
    
    print(f"✅ Janelas de dados geradas. Shape de X: {X.shape}, Shape de y: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, shuffle=False # shuffle=False é melhor para séries temporais
    )

    # O input_shape agora tem +1 feature (o nível do rio anterior)
    num_features = len(features_chuva) + 1
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(NUM_LAGS, num_features)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    
    print("🧠  Iniciando o treinamento do modelo LSTM...")
    model.fit(
        X_train, y_train, 
        epochs=50, 
        batch_size=16, 
        validation_data=(X_test, y_test), # Usar o conjunto de teste para validação
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
        verbose=1
    )

    print("💾  Salvando o modelo e os artefatos...")
    model.save('models/lstm_model.keras')
    joblib.dump(scaler, 'models/scaler.pkl')
    # Não precisamos mais salvar a lista de afluentes, o scaler cuida disso
    print("✅ Modelo e artefatos salvos com sucesso!")

if __name__ == '__main__':
    train_model()