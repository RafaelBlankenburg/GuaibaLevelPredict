# Arquivo: main.py

from src.data_collection import coletar_dados_chuva, coletar_nivel_atual_rio
from src.prediction import prever_nivel_rio_sequencia

# --- CONFIGURAÇÕES GERAIS DA EXECUÇÃO ---
# Estas constantes controlam a execução da previsão.
NUM_DIAS_HISTORICO = 14
NUM_DIAS_PREVISAO = 14
NUM_LAGS_MODELO = 14 # Deve ser o mesmo NUM_LAGS do seu script de treino

def run_prediction_pipeline():
    """
    Orquestra a execução completa da pipeline de previsão.
    """
    print("--- INICIANDO ROTINA DE PREVISÃO DO NÍVEL DO RIO GUAIBA ---")
    
    # PASSO 1: Coletar todos os dados de entrada
    df_chuva_total = coletar_dados_chuva(NUM_DIAS_HISTORICO, NUM_DIAS_PREVISAO)
    nivel_atual = coletar_nivel_atual_rio()
    
    # PASSO 2: Executar a previsão em sequência com os dados coletados
    previsoes_finais = prever_nivel_rio_sequencia(
        chuva_df=df_chuva_total,
        nivel_inicial_rio=nivel_atual,
        num_dias_historico=NUM_DIAS_HISTORICO,
        num_dias_previsao=NUM_DIAS_PREVISAO,
        num_lags_modelo=NUM_LAGS_MODELO
    )
    
    # PASSO 3: Salvar e exibir os resultados
    previsoes_finais.to_csv("data/previsao_nivel_rio.csv", index=False)
    print("\n✅ Previsões salvas em data/previsao_nivel_rio.csv")
    
    print("\n📈 Previsões do nível do rio para os próximos dias:\n")
    previsoes_finais['altura_prevista_m'] = previsoes_finais['altura_prevista_m'].round(2)
    print(previsoes_finais.to_string(index=False))


if __name__ == "__main__":
    # O TREINAMENTO NÃO É EXECUTADO AQUI.
    # Para treinar um novo modelo, execute o script: python src/train.py
    
    run_prediction_pipeline()