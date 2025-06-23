import pandas as pd
import numpy as np
import os 
from src.data_collection import coletar_dados_chuva, coletar_nivel_atual_rio
from src.prediction import prever_nivel_rio_sequencia
from src.visualization import gerar_grafico_previsao 

# --- CONFIGURAÇÕES GERAIS DA EXECUÇÃO ---
NUM_DIAS_HISTORICO = 14
NUM_LAGS_MODELO = 14
NUM_DIAS_PREVISAO_CHUVA = 14
DIAS_ADICIONAIS_ESTIMATIVA = 7
DIAS_TOTAIS_PREVISAO = NUM_DIAS_PREVISAO_CHUVA + DIAS_ADICIONAIS_ESTIMATIVA
COTA_INUNDACAO = 3


def run_prediction_scenarios():
    """
    Orquestra a execução da pipeline, baseada em cenários e com separação
    visual entre previsão e estimativa.
    """
    print("--- INICIANDO ROTINA DE PREVISÃO E ESTIMATIVA DO NÍVEL DO RIO ---")
    
    os.makedirs('results', exist_ok=True) 

    df_chuva_base = coletar_dados_chuva(NUM_DIAS_HISTORICO, NUM_DIAS_PREVISAO_CHUVA)
    nivel_atual = coletar_nivel_atual_rio()
    
    print("\n---  сценарио 1: PREVISÃO COM ESTIAGEM (SEM CHUVA APÓS D14) ---")
    df_chuva_cenario1 = df_chuva_base.copy()

    if DIAS_ADICIONAIS_ESTIMATIVA > 0:
        datas_futuras = pd.to_datetime(pd.date_range(start=df_chuva_cenario1.index[-1] + pd.Timedelta(days=1), periods=DIAS_ADICIONAIS_ESTIMATIVA, freq='D'))
        df_zeros = pd.DataFrame(0, index=datas_futuras, columns=df_chuva_cenario1.columns)
        df_chuva_cenario1 = pd.concat([df_chuva_cenario1, df_zeros])


    previsao_total_numerica = prever_nivel_rio_sequencia(
        chuva_df=df_chuva_cenario1,
        nivel_inicial_rio=nivel_atual,
        num_dias_historico=NUM_DIAS_HISTORICO,
        num_dias_previsao=DIAS_TOTAIS_PREVISAO,
        num_lags_modelo=NUM_LAGS_MODELO
    )
    previsao_total_numerica.rename(columns={'altura_prevista_m': 'nivel_m'}, inplace=True)

    # --- ### NOVO: CHAMADA PARA GERAR O GRÁFICO ### ---
    gerar_grafico_previsao(
        df_previsao=previsao_total_numerica,
        ponto_de_corte=NUM_DIAS_PREVISAO_CHUVA,
        cota_inundacao=COTA_INUNDACAO,
        caminho_saida='results/previsao_nivel_rio.png'
    )
    
    # --- Preparação para o print no console (com o separador de texto) ---
    df_previsao_real = previsao_total_numerica.iloc[:NUM_DIAS_PREVISAO_CHUVA]
    df_estimativa = previsao_total_numerica.iloc[NUM_DIAS_PREVISAO_CHUVA:]
    
    linha_separadora = pd.DataFrame([['--- Início da Estimativa ---', '---']], columns=previsao_total_numerica.columns)
    df_resultado_final_texto = pd.concat([df_previsao_real, linha_separadora, df_estimativa]).reset_index(drop=True)
    
    # Salva o CSV
    df_resultado_final_texto.to_csv("results/previsao_nivel_rio_com_estimativa.csv", index=False)
    print("\n✅ Previsões e estimativas em texto salvas em results/previsao_nivel_rio_com_estimativa.csv")
    
    print(f"\n📈 Previsões para {NUM_DIAS_PREVISAO_CHUVA} dias e Estimativas para mais {DIAS_ADICIONAIS_ESTIMATIVA} dias:\n")
    
    coluna_numerica = pd.to_numeric(df_resultado_final_texto['nivel_m'], errors='coerce').round(2)
    df_resultado_final_texto['nivel_m'] = coluna_numerica.fillna('---')
            
    print(df_resultado_final_texto.to_string(index=False))


if __name__ == "__main__":
    run_prediction_scenarios()