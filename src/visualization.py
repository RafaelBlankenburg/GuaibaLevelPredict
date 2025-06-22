import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def gerar_grafico_previsao(df_previsao, ponto_de_corte, cota_inundacao, caminho_saida):
    """
    Gera e salva um gráfico de linha com a previsão e a estimativa do nível do rio.

    Args:
        df_previsao (pd.DataFrame): DataFrame contendo os dados numéricos da previsão.
                                    Deve ter a coluna 'data' e a coluna do nível.
        ponto_de_corte (int): O índice que separa a previsão da estimativa.
        cota_inundacao (float): O nível da cota de inundação para plotar como referência.
        caminho_saida (str): O caminho do arquivo para salvar o gráfico (ex: 'results/previsao.png').
    """
    print("📈 Gerando gráfico da previsão...")

    # Garante que a coluna de data está no formato correto
    df_previsao['data'] = pd.to_datetime(df_previsao['data'])
    coluna_nivel = df_previsao.columns[1] # Pega o nome da segunda coluna (nível)

    # Separa os dados para plotar com estilos diferentes
    df_real_forecast = df_previsao.iloc[:ponto_de_corte]
    # Para a linha tracejada, incluímos o último ponto da previsão real para uma conexão suave
    df_estimate = df_previsao.iloc[ponto_de_corte-1:]

    # --- Criação do Gráfico ---
    plt.style.use('seaborn-v0_8-whitegrid') # Estilo visual agradável
    fig, ax = plt.subplots(figsize=(15, 8))

    # Plota a parte da previsão (com dados de chuva reais)
    ax.plot(df_real_forecast['data'], df_real_forecast[coluna_nivel], 
            'o-', label='Previsão (baseada na chuva prevista)', color='royalblue')

    # Plota a parte da estimativa (cenário sem chuva)
    ax.plot(df_estimate['data'], df_estimate[coluna_nivel], 
            'o--', label='Estimativa (cenário sem chuva futura)', color='darkorange')

    # Plota a linha da cota de inundação
    ax.axhline(y=cota_inundacao, color='red', linestyle='--', 
               label=f'Cota de Inundação ({cota_inundacao:.2f} m)')

    # --- Formatação e Rótulos ---
    ax.set_title('Previsão e Estimativa do Nível do Rio Guaíba', fontsize=16)
    ax.set_xlabel('Data', fontsize=12)
    ax.set_ylabel('Nível (metros)', fontsize=12)
    
    # Formata o eixo X para mostrar as datas de forma inteligente
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right') # Rotaciona as datas

    ax.legend()
    fig.tight_layout() # Ajusta o layout para evitar que os rótulos se sobreponham

    # --- Salvando o Gráfico ---
    plt.savefig(caminho_saida, dpi=150)
    plt.close(fig) # Fecha a figura para liberar memória
    
    print(f"✅ Gráfico salvo em: {caminho_saida}")