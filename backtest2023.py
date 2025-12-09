import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from src.ai_engine import PrevisorNivelRio
from src.data_loader import coletar_dados_historicos_arquivo
from src.config import CAMINHO_MODELO, DIAS_ATRASO_MODELO

# --- CONFIGURAÇÕES PARA SETEMBRO DE 2023 ---
# A enchente histórica ocorreu entre 02/09 e 08/09
ARQUIVO_REAIS = 'data/raw/niveis_reais_diarios.csv'
DATA_INICIO_SIMULACAO = "2023-08-20" # Começamos antes da chuva para pegar o nível baixo
DATA_FIM_SIMULACAO = "2023-09-20"    # Vamos até o fim do mês para ver a descida

def run_backtest_2023():
    print(f"--- 🔙 INICIANDO BACKTEST: SETEMBRO DE 2023 ---")

    # 1. Carregar Dados Reais (Gabarito)
    if not os.path.exists(ARQUIVO_REAIS):
        print(f"❌ Erro: Arquivo {ARQUIVO_REAIS} não encontrado.")
        return

    df_real = pd.read_csv(ARQUIVO_REAIS)
    df_real['data'] = pd.to_datetime(df_real['data'])
    df_real.set_index('data', inplace=True)
    
    # Filtra apenas o período de interesse para o gráfico ficar limpo
    try:
        df_recorte_real = df_real.loc[DATA_INICIO_SIMULACAO:DATA_FIM_SIMULACAO]
    except KeyError:
        print("⚠️ AVISO: Não encontrei dados de 2023 no seu CSV de níveis reais!")
        print("   O gráfico terá apenas a linha vermelha (IA), sem comparação.")
        df_recorte_real = pd.DataFrame() # Vazio

    if not df_recorte_real.empty:
        print(f"📊 Dados reais carregados: {len(df_recorte_real)} dias.")
        nivel_inicial = df_recorte_real.iloc[0]['altura_rio_guaiba_m']
    else:
        # Se não tiver dados reais, chutamos um nível inicial médio de inverno
        nivel_inicial = 1.20 
        print(f"⚠️ Usando nível inicial estimado: {nivel_inicial}m")

    # 2. Buscar Chuva Histórica (API Archive)
    # Precisamos de 90 dias ANTES do início da simulação para encher a memória da IA
    data_inicio_chuva = pd.to_datetime(DATA_INICIO_SIMULACAO) - pd.Timedelta(days=90)
    
    str_inicio_chuva = data_inicio_chuva.strftime('%Y-%m-%d')
    str_fim_chuva = DATA_FIM_SIMULACAO

    print(f"⏳ Baixando chuva histórica de {str_inicio_chuva} até {str_fim_chuva}...")
    df_chuva = coletar_dados_historicos_arquivo(str_inicio_chuva, str_fim_chuva)
    
    if df_chuva.empty:
        print("❌ Falha ao obter dados de chuva.")
        return

    # 3. Carregar Modelo
    ia = PrevisorNivelRio(dias_atraso=DIAS_ATRASO_MODELO)
    if not ia.carregar(CAMINHO_MODELO):
        print("❌ Erro: Modelo não encontrado. Treine primeiro (main.py).")
        return

    # 4. Executar Simulação
    print(f"🔮 Simulando comportamento de 2023...")
    
    df_simulacao = ia.prever_simulacao(df_chuva, nivel_inicial)
    
    # Corta para o período visual
    df_simulacao = df_simulacao.loc[DATA_INICIO_SIMULACAO:DATA_FIM_SIMULACAO]

    # 5. Gerar Gráfico
    plt.figure(figsize=(14, 7))
    
    # Plot IA
    plt.plot(df_simulacao.index, df_simulacao['nivel_estimado'], 
             color='#d62728', linewidth=2.5, marker='o', markersize=4, label='IA (Simulado)')
    
    # Plot Real (se existir)
    if not df_recorte_real.empty:
        plt.plot(df_recorte_real.index, df_recorte_real['altura_rio_guaiba_m'], 
                 color='black', linewidth=3, label='Real (Medido)')
        
        # Calcula Erro no Pico (opcional)
        pico_real = df_recorte_real['altura_rio_guaiba_m'].max()
        pico_ia = df_simulacao['nivel_estimado'].max()
        erro = pico_ia - pico_real
        print(f"\n📏 Comparativo de Pico:")
        print(f"   Real: {pico_real:.2f}m")
        print(f"   IA:   {pico_ia:.2f}m")
        print(f"   Diferença: {erro:+.2f}m")

    plt.title('Backtest: Enchente de Setembro 2023', fontsize=16, fontweight='bold')
    plt.xlabel('Data')
    plt.ylabel('Nível (m)')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    caminho_grafico = 'data/processed/backtest_2023_setembro.png'
    plt.savefig(caminho_grafico)
    print(f"\n✅ Gráfico salvo em: {caminho_grafico}")

if __name__ == "__main__":
    run_backtest_2023()