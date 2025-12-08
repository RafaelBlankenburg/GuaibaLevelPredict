import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import os

def plotar_previsao(df_previsao, nivel_atual, nome_arquivo='grafico_previsao.png'):
    """
    Gera um gráfico da previsão e salva na pasta data/processed.
    """
    # Configuração do tamanho da figura
    plt.figure(figsize=(12, 6))
    
    # Prepara os dados
    datas = df_previsao.index
    niveis = df_previsao['nivel_estimado']
    
    # 1. Plotar a Previsão (Linha Azul)
    plt.plot(datas, niveis, marker='o', linestyle='-', color='#007acc', linewidth=2, label='Previsao IA')
    
    # 2. Plotar o Ponto Atual (Ponto Verde)
    data_inicial = datas[0] - pd.Timedelta(days=1)
    plt.scatter([data_inicial], [nivel_atual], color='green', s=120, zorder=5, label=f'Nivel Atual ({nivel_atual}m)')
    
    # Linha tracejada conectando o atual à primeira previsão
    plt.plot([data_inicial, datas[0]], [nivel_atual, niveis.iloc[0]], color='green', linestyle='--', alpha=0.6)

    # 3. Linhas de Referência (Cotas do Guaíba)
    plt.axhline(y=2.50, color='orange', linestyle='--', alpha=0.7, label='Alerta (2.5m)')
    plt.axhline(y=3.00, color='red', linestyle='--', alpha=0.7, label='Inundacao (3.0m)')

    # Formatação do Gráfico (Sem acentos ou emojis para evitar erros de fonte)
    plt.title('Previsao do Nivel do Rio Guaiba', fontsize=16, fontweight='bold')
    plt.xlabel('Data', fontsize=12)
    plt.ylabel('Nivel (metros)', fontsize=12)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    
    # Formatação das Datas no Eixo X
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
    plt.xticks(rotation=45)
    
    plt.legend(loc='upper left')
    plt.tight_layout()

    # Salvar arquivo
    caminho_saida = os.path.join('data', 'processed', nome_arquivo)
    os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)
    
    plt.savefig(caminho_saida, dpi=100)
    print(f"📈 Grafico gerado com sucesso em: {caminho_saida}")