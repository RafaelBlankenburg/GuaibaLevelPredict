import pandas as pd
import requests_cache
from retry_requests import retry
from datetime import datetime

PERIODO_INICIO = "2023-08-01"
PERIODO_FIM = "2023-12-09"
NOME_ARQUIVO_SAIDA = "rain.csv"


DADOS_NIVEL_RIO_M = [
    1.23,     1.19,     1.40,     1.25,     1.30,     1.15,     1.51,    
    1.48,     1.23,     1.19,     1.17,     1.43,     1.42,     1.19,    
    1.12,     1.08,     1.13,     1.25,     1.32,     1.14,     1.16,    
    1.09,     1.30,     1.34,     1.24,     1.36,     1.34,     1.25,    
    1.19,     1.06,     1.16,     1.15,     0.98,     0.99,     1.64,    
    1.90,     2.33,     2.47,     2.46,     2.32,     2.20,     2.05,    
    2.20,     2.45,     2.70,     2.55,     2.53,     2.54,     2.62,    
    2.68,     2.63,     2.64,     2.68,     2.66,     2.73,     2.76,    
    2.80,     3.18,     3.02,     2.79,     2.84,     2.78,     2.63,    
    2.53,     2.47,     2.47,     2.47,     2.47,     2.47,     2.58,    
    2.53,     2.53,     2.52,     2.46,     2.46,     2.36,     2.12,    
    2.35,     2.43,     2.43,     2.35,     2.21,     2.04,     1.99,    
    2.17,     2.16,     2.07,     1.93,     1.76,     1.68,     1.69,    
    1.69,     1.69,     1.69,     1.64,     1.87,     1.89,     1.77,    
    1.60,     1.46,     1.62,     1.48,     1.42,     1.56,     1.60,    
    1.79,     1.84,     1.97,     2.02,     2.24,     2.76,     3.38,    
    3.46,     3.32,     3.09,     2.78,     2.74,     2.52,     2.30,    
    2.18,     2.10,     2.12,     2.11,     1.98,     2.01,     2.11,    
    1.90,     1.64,     1.88,     1.98,     1.74
]


# --- 3. CIDADES (NÃO PRECISA EDITAR) ---
CIDADES = {
    "vacaria_mm": (-28.5108, -50.9389), "guapore_mm": (-28.8489, -51.8903),
    "lagoa_vermelha_mm": (-28.2086, -51.5258), "passo_fundo_mm": (-28.2628, -52.4067),
    "soledade_mm": (-28.8189, -52.5103), "cruz_alta_mm": (-28.6389, -53.6064),
    "salto_do_jacui_mm": (-29.0950, -53.2144), "sao_francisco_de_paula_mm": (-29.4481, -50.5822),
    "bento_goncalves_mm": (-29.1686, -51.5189), "caxias_do_sul_mm": (-29.1678, -51.1789),
    "lajeado_mm": (-29.4669, -51.9614), "taquari_mm": (-29.7983, -51.8619),
    "santa_cruz_do_sul_mm": (-29.7178, -52.4258), "julio_de_castilhos_mm": (-29.2269, -53.6817),
    "santa_maria_mm": (-29.6842, -53.8069), "viamao_mm": (-30.0811, -51.0233),
    "cachoeira_do_sul_mm": (-30.0392, -52.8939), "encruzilhada_do_sul_mm": (-30.5428, -52.5219),
    "cacapava_do_sul_mm": (-30.5128, -53.4911), "sao_gabriel_mm": (-30.3358, -54.3200)
}

def montar_base_historica():
    """
    Função principal que orquestra a coleta e montagem do dataset.
    """
    print(f"--- Iniciando montagem da base de dados para o período de {PERIODO_INICIO} a {PERIODO_FIM} ---")

    start_date = datetime.strptime(PERIODO_INICIO, "%Y-%m-%d")
    end_date = datetime.strptime(PERIODO_FIM, "%Y-%m-%d")
    num_dias = (end_date - start_date).days + 1

    if len(DADOS_NIVEL_RIO_M) != num_dias:
        raise ValueError(
            f"Erro: O número de dias no período ({num_dias}) é diferente do "
            f"número de entradas na lista de níveis do rio ({len(DADOS_NIVEL_RIO_M)})."
        )

    print("⏳ Coletando dados de chuva para todas as cidades de uma vez...")
    
    latitudes = [coord[0] for coord in CIDADES.values()]
    longitudes = [coord[1] for coord in CIDADES.values()]
    
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    
    params = {
        "latitude": latitudes, "longitude": longitudes,
        "start_date": PERIODO_INICIO, "end_date": PERIODO_FIM,
        "daily": "rain_sum", "timezone": "America/Sao_Paulo"
    }
    
    url = "https://archive-api.open-meteo.com/v1/archive"
    response = retry_session.get(url, params=params, timeout=60)
    response.raise_for_status()
    
    data = response.json()
    print("✅ Dados de chuva coletados com sucesso!")

    chuva_data = {}
    for i, nome_cidade in enumerate(CIDADES.keys()):
        chuva_data[nome_cidade] = data[i]['daily']['rain_sum']
    
    df_chuva = pd.DataFrame(chuva_data)
    df_chuva["altura_rio_guaiba_m"] = DADOS_NIVEL_RIO_M
    
    df_chuva.index = pd.to_datetime(data[0]['daily']['time'])

    df_chuva.to_csv(NOME_ARQUIVO_SAIDA, index=False) 
    
    print(f"\n✅ Arquivo '{NOME_ARQUIVO_SAIDA}' gerado com sucesso (sem a coluna de data)!")
    print("Amostra do resultado:")
    print(df_chuva.head())

if __name__ == "__main__":
    montar_base_historica()