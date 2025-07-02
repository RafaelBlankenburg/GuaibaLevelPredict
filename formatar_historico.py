import pandas as pd

NOME_ARQUIVO_CSV_ENTRADA = 'dados_brutos_nivel.csv'
NOME_ARQUIVO_CSV_SAIDA = 'niveis_reais_diarios.csv'

def formatar_dados():
    print(f"🔄 Carregando dados do arquivo: '{NOME_ARQUIVO_CSV_ENTRADA}'...")
    try:
        df = pd.read_csv(NOME_ARQUIVO_CSV_ENTRADA, header=None, names=['timestamp', 'nivel_cm'])
    except FileNotFoundError:
        print(f"ERRO: Arquivo '{NOME_ARQUIVO_CSV_ENTRADA}' não encontrado.")
        return

    df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True)
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)

    df_diario_max = df['nivel_cm'].resample('D').max()
    df_diario_max.dropna(inplace=True)
    df_em_metros = df_diario_max / 100

    df_em_metros.to_csv(NOME_ARQUIVO_CSV_SAIDA, index_label='data', header=['altura_rio_guaiba_m'])
    print(f"✅ Processamento Concluído! Os dados diários foram salvos em '{NOME_ARQUIVO_CSV_SAIDA}'")

if __name__ == "__main__":
    formatar_dados()