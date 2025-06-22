import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.preprocess import gerar_janelas_lstm

# --- CONSTANTES DO MODELO ---
NUM_LAGS = 7
RANDOM_SEED = 42
# --- ALTERAÇÃO 1: Nome do arquivo atualizado para refletir os dados que estamos usando ---
ARQUIVO_CSV = 'data/rain.csv' # Use o nome do seu arquivo de dados reais

def train_model():
    """
    Função para treinar o modelo LSTM com os dados históricos de chuva e nível do rio.
    """
    print("⚙️  Iniciando treinamento do modelo com dados históricos...")
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    # Carregar dados reais
    df = pd.read_csv(ARQUIVO_CSV)

    # --- ALTERAÇÃO 2: Definição dinâmica e correta das colunas de features e alvo ---
    # Define qual é a coluna que queremos prever (nosso alvo ou 'target')
    COLUNA_ALVO = 'altura_rio_guaiba_m'
    
    # As features (afluentes) são todas as outras colunas que não são o alvo.
    # Este método é robusto e se adapta caso você adicione ou remova cidades no futuro.
    afluentes = [col for col in df.columns if col != COLUNA_ALVO]
    
    print(f"✅ Colunas de features ('afluentes') identificadas: {len(afluentes)} colunas.")
    print(f"✅ Coluna alvo ('target') identificada: {COLUNA_ALVO}")

    # Padronizar apenas as colunas de features (afluentes)
    scaler = StandardScaler()
    df[afluentes] = scaler.fit_transform(df[afluentes])
    
    # --- OBSERVAÇÃO IMPORTANTE SOBRE A FUNÇÃO ABAIXO ---
    # Sua função 'gerar_janelas_lstm' deve receber o dataframe e o número de lags
    # e saber internamente que as colunas em 'afluentes' são as features (X) e
    # a 'COLUNA_ALVO' é o que será usado para gerar os rótulos (y).
    X, y = gerar_janelas_lstm(df, NUM_LAGS, COLUNA_ALVO, afluentes)

    print(f"✅ Janelas de dados geradas. Shape de X: {X.shape}, Shape de y: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )

    # --- A definição do modelo agora usa 'len(afluentes)' que está correto ---
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=True, input_shape=(NUM_LAGS, len(afluentes))),
        tf.keras.layers.Dropout(0.2), # Adicionado Dropout para regularização
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dropout(0.2), # Adicionado Dropout para regularização
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse', metrics=['mae']) # Adicionada métrica MAE
    
    print("🧠  Iniciando o treinamento do modelo LSTM...")
    history = model.fit(
        X_train, y_train, 
        epochs=50, # Aumentei as épocas, 30 pode ser pouco
        batch_size=16, 
        validation_split=0.2,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)] # Parada antecipada
    )

    print("📊  Avaliando o modelo no conjunto de teste...")
    loss, mae = model.evaluate(X_test, y_test)
    print(f'-> Erro Quadrático Médio (MSE) de Teste: {loss:.4f}')
    print(f'-> Erro Absoluto Médio (MAE) de Teste: {mae:.4f}')


    print("💾  Salvando o modelo e os artefatos...")
    model.save('models/lstm_model.keras')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(afluentes, 'models/afluentes.pkl') # Salva a lista correta de afluentes
    print("✅ Modelo e artefatos salvos com sucesso!")


# Adicionei uma sugestão para a sua função de preprocessamento
def gerar_janelas_lstm_sugestao(df, num_lags, coluna_alvo, colunas_features):
    """
    Função de exemplo para gerar janelas para o modelo LSTM.
    Adapte conforme a sua implementação real em 'src/preprocess.py'.
    """
    X, y = [], []
    for i in range(len(df) - num_lags):
        janela_x = df[colunas_features].iloc[i:i + num_lags].values
        valor_y = df[coluna_alvo].iloc[i + num_lags]
        X.append(janela_x)
        y.append(valor_y)
    return np.array(X), np.array(y)

if __name__ == '__main__':
    # Para que o código funcione, sua função em src/preprocess.py precisa estar alinhada.
    # Vou usar a função de sugestão aqui para demonstração.
    # No seu código, mantenha a importação original.
    from src.preprocess import gerar_janelas_lstm 
    train_model()