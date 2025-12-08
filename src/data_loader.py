import pandas as pd
import numpy as np
import openmeteo_requests
import requests_cache
from retry_requests import retry
from src.config import CIDADES

def carregar_dados_treino(caminho_csv):
    """Carrega o dataset histórico local."""
    try:
        df = pd.read_csv(caminho_csv)
        print(f"📂 Dados históricos carregados: {len(df)} registros.")
        return df
    except FileNotFoundError:
        print(f"❌ Erro: Arquivo {caminho_csv} não encontrado.")
        return None

def _get_openmeteo_client():
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    return openmeteo_requests.Client(session=retry(cache_session, retries=5, backoff_factor=0.2))

def coletar_dados_chuva_api(dias_historico, dias_previsao):
    """Consulta a API para obter dados recentes e futuros."""
    print(f"⏳ Coletando dados da API ({dias_historico} dias passados + {dias_previsao} dias futuros)...")
    
    openmeteo = _get_openmeteo_client()
    dfs = []
    all_dates = None

    for nome_cidade, lat, lon in CIDADES:
        params = {
            "latitude": lat, "longitude": lon, "daily": "rain_sum",
            "timezone": "America/Sao_Paulo", "past_days": dias_historico, "forecast_days": dias_previsao
        }
        try:
            responses = openmeteo.weather_api("https://api.open-meteo.com/v1/forecast", params=params)
            response = responses[0]
            daily = response.Daily()
            
            rain_values = daily.Variables(0).ValuesAsNumpy()
            df = pd.DataFrame(data=rain_values, columns=[nome_cidade])

            if all_dates is None:
                start_date = pd.to_datetime(daily.Time(), unit="s").date()
                all_dates = pd.date_range(start=start_date, periods=len(df), freq='D')
            
            dfs.append(df)
        except Exception as e:
            print(f"⚠️ Falha na cidade {nome_cidade}: {e}")
            # Em produção, idealmente tratar melhor, aqui preenchemos com 0 para não quebrar
            dfs.append(pd.DataFrame(np.zeros((dias_historico + dias_previsao, 1)), columns=[nome_cidade]))

    if not dfs:
        return pd.DataFrame()

    chuva_df = pd.concat(dfs, axis=1)
    if all_dates is not None:
        chuva_df = chuva_df.head(len(all_dates))
        chuva_df.index = all_dates
    
    return chuva_df

def coletar_dados_historicos_arquivo(data_inicio, data_fim):
    """
    Busca dados na API de ARQUIVO (Archive) para Backtest.
    data_inicio e data_fim devem ser strings 'YYYY-MM-DD'.
    """
    print(f"⏳ Buscando chuva histórica (Archive) de {data_inicio} a {data_fim}...")
    
    openmeteo = _get_openmeteo_client()
    url = "https://archive-api.open-meteo.com/v1/archive"
    
    # Prepara lista para receber DataFrames de cada cidade
    lista_dfs = []
    
    # Para garantir que o índice de datas seja único e correto, vamos criar o range esperado
    range_datas = pd.date_range(start=data_inicio, end=data_fim, freq='D')
    
    for nome_cidade, lat, lon in CIDADES:
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": data_inicio,
            "end_date": data_fim,
            "daily": "rain_sum",
            "timezone": "America/Sao_Paulo"
        }
        
        try:
            # Faz a requisição
            responses = openmeteo.weather_api(url, params=params)
            response = responses[0]
            daily = response.Daily()
            
            # Extrai os valores de chuva
            rain_values = daily.Variables(0).ValuesAsNumpy()
            
            # Cria DataFrame temporário para esta cidade
            # Nota: A API retorna datas unix, mas como já definimos start/end, 
            # podemos forçar o index para garantir alinhamento
            if len(rain_values) == len(range_datas):
                df_cidade = pd.DataFrame(rain_values, index=range_datas, columns=[nome_cidade])
            else:
                # Caso a API retorne tamanho diferente (raro), ajustamos
                print(f"⚠️ Tamanho diferente para {nome_cidade}. Tentando ajustar...")
                df_cidade = pd.DataFrame(data=rain_values, columns=[nome_cidade])
            
            lista_dfs.append(df_cidade)
            
        except Exception as e:
            print(f"❌ Falha ao buscar dados para {nome_cidade}: {e}")
            # Cria coluna de zeros para não quebrar
            lista_dfs.append(pd.DataFrame(0.0, index=range_datas, columns=[nome_cidade]))

    if not lista_dfs:
        return pd.DataFrame()

    # Junta todas as cidades lado a lado
    chuva_df = pd.concat(lista_dfs, axis=1)
    
    # Remove qualquer linha duplicada ou vazia
    chuva_df = chuva_df.loc[~chuva_df.index.duplicated(keep='first')]
    
    return chuva_df