import os
import pandas as pd
from src.ai_engine import PrevisorNivelRio
from src.data_loader import coletar_dados_chuva_api, carregar_dados_treino
from src.config import DIAS_ATRASO_MODELO, CAMINHO_MODELO
from src.visualization import plotar_previsao 

def main():
    print("--- SISTEMA DE PREVISAO DE NIVEL DE RIOS ---")
    
    # 1. Instancia o Modelo
    ia = PrevisorNivelRio(dias_atraso=DIAS_ATRASO_MODELO)
    
    # 2. Verifica/Carrega Modelo
    carregou = ia.carregar(CAMINHO_MODELO)
    
    if not carregou:
        print("⚠️ Nenhum modelo salvo encontrado. Iniciando treinamento...")
        arquivo_historico = 'data/raw/historico_completo.csv' 
        
        if not os.path.exists(arquivo_historico):
            print(f"❌ Arquivo {arquivo_historico} não encontrado.")
            return
        
        df_treino = carregar_dados_treino(arquivo_historico)
        ia.treinar(df_treino)
        ia.salvar(CAMINHO_MODELO)
    else:
        print("✅ Modelo carregado com sucesso.")

    # 3. Coleta dados API (AUMENTADO AQUI)
    # dias_historico=30: Pega 1 mês para trás para garantir que os lags estejam preenchidos
    # dias_previsao=16: O máximo que a API gratuita costuma entregar
    df_api = coletar_dados_chuva_api(dias_historico=30, dias_previsao=16)
    
    if df_api.empty:
        print("❌ Falha ao obter dados da API.")
        return

    # 4. Nível Atual
    nivel_atual_hoje = 0.77 
    print(f"📏 Nivel atual considerado: {nivel_atual_hoje}m")

    # 5. Executa a Previsão e Gera Gráfico
    try:
        previsoes = ia.prever_simulacao(df_api, nivel_atual_hoje)
        print("\n📊 RESULTADO DA PREVISAO (Proximos dias):")
        print(previsoes) # Vai mostrar todos os dias
        
        # Salva CSV
        previsoes.to_csv('data/processed/previsao_atual.csv')
        
        # Gera o gráfico
        plotar_previsao(previsoes, nivel_atual_hoje)
        
    except Exception as e:
        print(f"❌ Erro durante a execucao: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()