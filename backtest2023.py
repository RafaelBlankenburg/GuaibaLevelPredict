import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from src.ai_engine import PrevisorNivelRio
from src.data_loader import coletar_dados_historicos_arquivo
from src.config import CAMINHO_MODELO, DIAS_ATRASO_MODELO

# --- CONFIGURAÇÕES ---
ARQUIVO_REAIS = 'data/raw/nivel_reais_diarios2.csv' # Seu arquivo convertido
DATA_INICIO_SIMULACAO = "2023-08-20" 
DATA_FIM_SIMULACAO = "2023-09-30"    

def run_backtest_2023():
    print(f"--- 🔙 INICIANDO BACKTEST: SETEMBRO DE 2023 ---")

    # 1. Carregar Dados Reais
    if not os.path.exists(ARQUIVO_REAIS):
        print(f"❌ Erro: Arquivo {ARQUIVO_REAIS} não encontrado.")
        return

    df_real = pd.read_csv(ARQUIVO_REAIS)
    df_real['data'] = pd.to_datetime(df_real['data'])
    df_real.set_index('data', inplace=True)
    
    try:
        # Pega o nível real EXATAMENTE no dia de início
        # Se não tiver dado nesse dia exato, pega o anterior mais próximo (method='ffill')
        idx_inicio = pd.Timestamp(DATA_INICIO_SIMULACAO)
        
        # Garante que temos dados suficientes no real para encontrar o ponto de partida
        df_real_sorted = df_real.sort_index()
        nivel_inicial = df_real_sorted.asof(idx_inicio)['altura_rio_guaiba_m']
        
        print(f"📍 Nível Real de Partida ({idx_inicio.date()}): {nivel_inicial}m")
        
    except Exception as e:
        print(f"⚠️ Erro ao buscar nível inicial: {e}")
        nivel_inicial = 1.00 # Fallback

    # 2. Buscar Chuva (O TRUQUE DO CORTE PERFEITO)
    # O engine descarta os primeiros 35 dias (buffer).
    # Então baixamos (Data Inicio - 35 dias).
    # Assim, o dia 36 (o primeiro da previsão) cai EXATAMENTE na Data Inicio.
    buffer_engine = 35 
    data_corte_inicio = pd.Timestamp(DATA_INICIO_SIMULACAO) - pd.Timedelta(days=buffer_engine)
    
    str_inicio_chuva = data_corte_inicio.strftime('%Y-%m-%d')
    str_fim_chuva = DATA_FIM_SIMULACAO

    print(f"⏳ Baixando chuva de {str_inicio_chuva} (Buffer) até {str_fim_chuva}...")
    df_chuva = coletar_dados_historicos_arquivo(str_inicio_chuva, str_fim_chuva)
    
    if df_chuva.empty: return

    # Garante que o dataframe de chuva comece EXATAMENTE no dia do corte
    # Isso alinha o índice 35 do engine com o dia DATA_INICIO_SIMULACAO
    df_chuva = df_chuva[df_chuva.index >= data_corte_inicio]

    # 3. Carregar e Simular
    ia = PrevisorNivelRio(dias_atraso=DIAS_ATRASO_MODELO)
    if not ia.carregar(CAMINHO_MODELO): return

    print(f"🔮 Simulando...")
    # O engine vai "comer" os 35 dias de buffer e usar 'nivel_inicial' no dia 36
    df_simulacao = ia.prever_simulacao(df_chuva, nivel_inicial)
    
    # Recorta para visualização
    df_simulacao = df_simulacao.loc[DATA_INICIO_SIMULACAO:DATA_FIM_SIMULACAO]
    df_recorte_real = df_real.loc[DATA_INICIO_SIMULACAO:DATA_FIM_SIMULACAO]

    # 4. Gráfico
    plt.figure(figsize=(14, 7))
    
    plt.plot(df_simulacao.index, df_simulacao['nivel_estimado'], 
             color='#d62728', linewidth=2.5, marker='o', markersize=4, label='IA (Simulado)')
    
    if not df_recorte_real.empty:
        plt.plot(df_recorte_real.index, df_recorte_real['altura_rio_guaiba_m'], 
                 color='black', linewidth=3, label='Real (Medido)')

    plt.title(f'Backtest 2023: Partida Sincronizada em {DATA_INICIO_SIMULACAO}', fontsize=16)
    plt.ylabel('Nível (m)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
    plt.tight_layout()
    
    plt.savefig('data/processed/backtest_2023_sincronizado.png')
    print(f"\n✅ Gráfico salvo.")

if __name__ == "__main__":
    run_backtest_2023()