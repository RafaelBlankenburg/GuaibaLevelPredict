import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def gerar_grafico_previsao(df_previsao, ponto_de_corte, cota_inundacao, caminho_saida):
    print("📈 Gerando gráfico da previsão...")


    df_previsao['data'] = pd.to_datetime(df_previsao['data'])
    coluna_nivel = df_previsao.columns[1] 
    df_real_forecast = df_previsao.iloc[:ponto_de_corte]
    df_estimate = df_previsao.iloc[ponto_de_corte-1:]


    plt.style.use('seaborn-v0_8-whitegrid') 
    fig, ax = plt.subplots(figsize=(15, 8))

    ax.plot(df_real_forecast['data'], df_real_forecast[coluna_nivel], 
            'o-', label='Previsão (baseada na chuva prevista)', color='royalblue')

    ax.plot(df_estimate['data'], df_estimate[coluna_nivel], 
            'o--', label='Estimativa (cenário sem chuva futura)', color='darkorange')

    ax.axhline(y=cota_inundacao, color='red', linestyle='--', 
               label=f'Cota de Inundação ({cota_inundacao:.2f} m)')

    ax.set_title('Previsão e Estimativa do Nível do Rio Guaíba', fontsize=16)
    ax.set_xlabel('Data', fontsize=12)
    ax.set_ylabel('Nível (metros)', fontsize=12)

    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right') 

    ax.legend()
    fig.tight_layout() 


    plt.savefig(caminho_saida, dpi=150)
    plt.close(fig) 
    
    print(f"✅ Gráfico salvo em: {caminho_saida}")