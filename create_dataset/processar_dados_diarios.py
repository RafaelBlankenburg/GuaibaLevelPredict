import pandas as pd


NOME_ARQUIVO_CSV = 'seus_dados.csv'
PERIODO_INICIO = '2023-08-01'
PERIODO_FIM = '2023-12-09'


def formatar_dados_para_lista():
    """
    Lê um arquivo CSV, extrai o valor máximo diário para um período
    específico e imprime uma lista Python formatada.
    """
    print(f"🔄 Carregando dados do arquivo: '{NOME_ARQUIVO_CSV}'...")
    
    try:
        df = pd.read_csv(NOME_ARQUIVO_CSV, header=None, names=['timestamp', 'nivel_cm'])
    except FileNotFoundError:
        print(f"ERRO: Arquivo não encontrado! Verifique se o nome '{NOME_ARQUIVO_CSV}' está correto.")
        return

    print("⚙️  Processando os dados...")


    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d/%m/%Y %H:%M')


    df.set_index('timestamp', inplace=True)

    df_diario_max = df['nivel_cm'].resample('D').max()

    idx_completo = pd.date_range(start=PERIODO_INICIO, end=PERIODO_FIM, freq='D')
    df_final = df_diario_max.reindex(idx_completo)

    df_final.fillna(method='ffill', inplace=True)


    df_em_metros = df_final / 100

    lista_de_niveis = df_em_metros.tolist()
    
    print("\n" + "="*50)
    print("✅ Processamento Concluído!")
    print(f"A lista contém {len(lista_de_niveis)} valores, correspondendo ao período de {PERIODO_INICIO} a {PERIODO_FIM}.")
    print("COPIE O BLOCO DE CÓDIGO ABAIXO E COLE NO SEU SCRIPT 'gerar_dataset.py'")
    print("="*50 + "\n")

    print("DADOS_NIVEL_RIO_M = [")
    for i, nivel in enumerate(lista_de_niveis):
        print(f"    {nivel:.2f}, ", end="")
        if (i + 1) % 7 == 0:
            print()
    print("\n]")
    print("\n" + "="*50)


if __name__ == "__main__":
    formatar_dados_para_lista()