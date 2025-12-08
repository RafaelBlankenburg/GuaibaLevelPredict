import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
from src.ai_engine import PrevisorNivelRio
from src.data_loader import coletar_dados_historicos_arquivo
from src.config import CAMINHO_MODELO, DIAS_ATRASO_MODELO

# --- CONFIGURAÇÕES ---
ARQUIVO_REAIS = 'data/raw/niveis_reais_diarios.csv'

def run_backtest():
    print(f"--- 🔙 INICIANDO BACKTEST (Sincronizado com CSV Real) ---")

    # 1. Carregar Dados Reais
    if not os.path.exists(ARQUIVO_REAIS):
        print(f"❌ Erro: Arquivo {ARQUIVO_REAIS} não encontrado.")
        return

    df_real = pd.read_csv(ARQUIVO_REAIS)
    df_real['data'] = pd.to_datetime(df_real['data'])
    df_real.set_index('data', inplace=True)
    
    # Define o período exato baseado no CSV
    data_inicio_real = df_real.index.min()
    data_fim_real = df_real.index.max()
    
    print(f"📊 Período do Backtest: {data_inicio_real.date()} até {data_fim_real.date()}")

    # 2. Buscar Chuva (Recuando 30 dias para garantir os lags)
    data_inicio_chuva = data_inicio_real - pd.Timedelta(days=30)
    
    str_inicio = data_inicio_chuva.strftime('%Y-%m-%d')
    str_fim = data_fim_real.strftime('%Y-%m-%d')

    print(f"⏳ Baixando chuva histórica de {str_inicio} até {str_fim}...")
    df_chuva = coletar_dados_historicos_arquivo(str_inicio, str_fim)
    
    if df_chuva.empty:
        print("❌ Falha ao obter dados de chuva.")
        return

    # 3. Carregar Modelo
    ia = PrevisorNivelRio(dias_atraso=DIAS_ATRASO_MODELO)
    if not ia.carregar(CAMINHO_MODELO):
        print("❌ Erro: Modelo não treinado.")
        return

    # 4. Executar Simulação
    # Pegamos o primeiro nível real conhecido para iniciar a recursão
    nivel_inicial = df_real.iloc[0]['altura_rio_guaiba_m']
    print(f"🔮 Simulando a partir de {nivel_inicial}m...")
    
    df_simulacao = ia.prever_simulacao(df_chuva, nivel_inicial)
    
    # Corta a simulação para ficar exatamente no mesmo tamanho dos dados reais
    # Inner Join garante que só ficamos com as datas que existem nos dois
    df_final = df_real.join(df_simulacao, how='inner', lsuffix='_real', rsuffix='_ia')
    
    # Renomear colunas para clareza
    df_final.rename(columns={'altura_rio_guaiba_m': 'Nivel_Real', 'nivel_estimado': 'Nivel_IA'}, inplace=True)

    # 5. Gerar Gráfico Focado
    plt.figure(figsize=(12, 6))
    
    # Área preenchida entre as linhas para destacar o erro
    plt.fill_between(df_final.index, df_final['Nivel_Real'], df_final['Nivel_IA'], 
                     where=(df_final['Nivel_Real'] > df_final['Nivel_IA']),
                     color='red', alpha=0.1, label='Erro (Subestimação)')

    plt.plot(df_final.index, df_final['Nivel_IA'], 
             color='#d62728', linewidth=2, marker='o', markersize=4, label='IA (Simulado)')
    
    plt.plot(df_final.index, df_final['Nivel_Real'], 
             color='black', linewidth=3, label='Real (Medido)')
    
    plt.title('Backtest Focado: Início da Enchente 2024', fontsize=14, fontweight='bold')
    plt.xlabel('Data')
    plt.ylabel('Nível do Rio (m)')
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Formatação de data
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    caminho_grafico = 'data/processed/backtest_2024_focado.png'
    plt.savefig(caminho_grafico)
    print(f"\n✅ Gráfico salvo: {caminho_grafico}")
    
    # Exibe métricas de erro para o final do período
    ultimo_real = df_final['Nivel_Real'].iloc[-1]
    ultimo_ia = df_final['Nivel_IA'].iloc[-1]
    erro = ultimo_real - ultimo_ia
    print(f"\n🛑 Diferença no último dia ({data_fim_real.date()}):")
    print(f"   Real: {ultimo_real:.2f}m")
    print(f"   IA:   {ultimo_ia:.2f}m")
    print(f"   Erro: {erro:.2f}m a menos que o real.")

if __name__ == "__main__":
    run_backtest()