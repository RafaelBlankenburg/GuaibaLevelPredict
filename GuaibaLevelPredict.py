import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import joblib


NUM_CIDADES = 5
NUM_LAGS = 10  
RANDOM_SEED = 42


np.random.seed(RANDOM_SEED)
dias = 500
chuvas = np.random.rand(dias, NUM_CIDADES) * 20 


altura_rio = np.zeros(dias)
for i in range(NUM_LAGS, dias):
    atraso = sum(chuvas[i-j].sum() * (0.3 / (j+1)) for j in range(NUM_LAGS))
    altura_rio[i] = atraso + np.random.normal(0, 1)

df = pd.DataFrame(chuvas, columns=[f'cidade_{i+1}' for i in range(NUM_CIDADES)])
df['altura_rio'] = altura_rio


def gerar_lags(df, num_lags):
    df_lags = pd.DataFrame()
    for cidade in [col for col in df.columns if col.startswith('cidade_')]:
        for lag in range(num_lags):
            df_lags[f'{cidade}_lag{lag}'] = df[cidade].shift(lag)
    df_lags['altura_rio'] = df['altura_rio']
    return df_lags.dropna().reset_index(drop=True)

df_lags = gerar_lags(df, NUM_LAGS)

X = df_lags.drop(columns='altura_rio')
y = df_lags['altura_rio']


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=RANDOM_SEED)
mlp = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=1000, random_state=RANDOM_SEED)
mlp.fit(X_train, y_train)


y_pred = mlp.predict(X_test)
print(f'Erro quadrático médio: {mean_squared_error(y_test, y_pred):.2f}')


joblib.dump(mlp, 'mlp_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(X.columns.tolist(), 'features.pkl')

def prever_altura(chuvas_recentes: list[list[float]]):
    """
    chuvas_recentes: lista de listas com formato [ [chuva_cidade1_hoje, ..., chuva_cidade5_hoje],
                                                   [chuva_cidade1_ontem, ..., chuva_cidade5_ontem], ... ]
    """
    assert len(chuvas_recentes) == NUM_LAGS, f'Esperado {NUM_LAGS} dias de histórico.'
    assert len(chuvas_recentes[0]) == NUM_CIDADES, f'Esperado {NUM_CIDADES} cidades.'

    entrada = []
    for lag in range(NUM_LAGS):
        for cidade in range(NUM_CIDADES):
            entrada.append(chuvas_recentes[lag][cidade])

    entrada_df = pd.DataFrame([entrada], columns=joblib.load('features.pkl'))
    scaler = joblib.load('scaler.pkl')
    entrada_scaled = scaler.transform(entrada_df)

    modelo = joblib.load('mlp_model.pkl')
    predicao = modelo.predict(entrada_scaled)
    return predicao[0]


exemplo_input = [
    [15, 5, 10, 3, 0],  
    [10, 0, 5, 0, 2],   
    [3, 0, 0, 0, 0],   
]
print("Previsão da altura do rio:", prever_altura(exemplo_input))
