# Arquivo: src/train.py (modificado)

import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

NUM_LAGS = 14
RANDOM_SEED = 60
ARQUIVO_CSV = 'data/rain.csv'

# ### ALTERADO: Função de gerar janelas muito mais simples e poderosa ###
def gerar_janelas(df, num_lags, cols_features, col_alvo):
    """
    Gera janelas de dados.
    X = Histórico de N dias de TODAS as features (chuva E rio)
    y = Nível do rio no dia seguinte
    """
    X, y = [], []
    for i in range(num_lags, len(df)):
        # Pega a janela de N dias atrás de todas as features de entrada
        janela_x = df[cols_features].iloc[i - num_lags : i].values
        X.append(janela_x)
        
        # O alvo é o valor da coluna 'col_alvo' no dia 'i'
        valor_y = df[col_alvo].iloc[i]
        y.append(valor_y)
        
    return np.array(X), np.array(y)

def train_model():
    print("⚙️  Iniciando treinamento com HISTÓRICO DO RIO como feature...")
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    df = pd.read_csv(ARQUIVO_CSV)

    COLUNA_ALVO = 'altura_rio_guaiba_m'
    features_chuva = [col for col in df.columns if col != COLUNA_ALVO]
    
    # ### ALTERADO: As features de entrada agora incluem o próprio histórico do rio ###
    FEATURES_ENTRADA = features_chuva + [COLUNA_ALVO]
    
    # A lógica de scalers separados continua correta e importante!
    scaler_chuva = StandardScaler()
    scaled_chuva = scaler_chuva.fit_transform(df[features_chuva])

    scaler_nivel = StandardScaler()
    scaled_nivel = scaler_nivel.fit_transform(df[[COLUNA_ALVO]])

    # Recria o DataFrame padronizado
    df_scaled = pd.DataFrame(scaled_chuva, columns=features_chuva)
    df_scaled[COLUNA_ALVO] = scaled_nivel
    
    # ### ALTERADO: Chamada da nova função de janelas ###
    X, y = gerar_janelas(df_scaled, NUM_LAGS, FEATURES_ENTRADA, COLUNA_ALVO)
    
    print(f"✅ Janelas de dados geradas. Shape de X: {X.shape}, Shape de y: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, shuffle=False
    )

    # O número de features continua o mesmo, mas agora uma delas é o próprio rio
    num_features = len(FEATURES_ENTRADA)
    
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
        validation_data=(X_test, y_test),
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)],
        verbose=1
    )

    print("💾  Salvando o modelo e os artefatos...")
    model.save('models/lstm_model.keras')
    joblib.dump(scaler_chuva, 'models/scaler_chuva.pkl')
    joblib.dump(scaler_nivel, 'models/scaler_nivel.pkl')
    print("✅ Modelo e artefatos salvos com sucesso!")

if __name__ == '__main__':
    train_model()