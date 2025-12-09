import pandas as pd
import numpy as np
import openmeteo_requests
import requests_cache
from retry_requests import retry
from src.config import CIDADES

def carregar_dados_treino(caminho_csv):
    try:
        df = pd.read_csv(caminho_csv)
        print(f"📂 Dados históricos carregados: {len(df)} registros.")
        return df
    except FileNotFoundError:
        print(f"❌ Erro: Arquivo {caminho_csv} não encontrado.")
        return None

def _get_openmeteo_client():
    # Cache de 1 hora para evitar bloqueios, mas permite atualizações frequentes
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    return openmeteo_requests.Client(session=retry(cache_session, retries=5, backoff_factor=0.2))

def coletar_dados_chuva_api(dias_historico, dias_previsao):
    """
    Busca dados recentes e previsão futura.
    Usa 'precipitation_sum' para garantir que pegamos todo tipo de água.
    """
    print(f"⏳ Coletando dados (Precipitação Total) - Histórico: {dias_historico}d | Futuro: {dias_previsao}d")
    
    openmeteo = _get_openmeteo_client()
    url = "https://api.open-meteo.com/v1/forecast"
    lista_dfs = []
    
    for nome_cidade, lat, lon in CIDADES:
        params = {
            "latitude": lat,
            "longitude": lon,
            # MUDANÇA 1: precipitation_sum pega chuva, garoa e granizo
            "daily": "precipitation_sum", 
            "timezone": "America/Sao_Paulo",
            "past_days": dias_historico,
            "forecast_days": dias_previsao
        }
        
        try:
            responses = openmeteo.weather_api(url, params=params)
            response = responses[0]
            daily = response.Daily()
            
            # Extrai valores
            rain_values = daily.Variables(0).ValuesAsNumpy()
            
            # --- MUDANÇA 2 (CRÍTICA): Adicionado () após .date ---
            start_ts = pd.to_datetime(daily.Time(), unit="s", utc=True).tz_convert("America/Sao_Paulo")
            start_date = start_ts.date() # <--- AQUI ESTAVA O ERRO
            
            dates = pd.date_range(start=start_date, periods=len(rain_values), freq='D')
            
            df = pd.DataFrame(rain_values, index=dates, columns=[nome_cidade])
            lista_dfs.append(df)
            
        except Exception as e:
            print(f"⚠️ Falha na cidade {nome_cidade}: {e}")
            pass

    if not lista_dfs:
        return pd.DataFrame()

    # Junta tudo (Outer Join para alinhar datas)
    chuva_df = pd.concat(lista_dfs, axis=1)
    
    # Preenche buracos com 0.0
    chuva_df.fillna(0.0, inplace=True)
    
    # Diagnóstico rápido no terminal
    hoje_str = pd.Timestamp.now().strftime('%Y-%m-%d')
    chuva_futura = chuva_df[chuva_df.index >= hoje_str]
    total_medio_futuro = chuva_futura.mean(axis=1).sum() if not chuva_futura.empty else 0
    print(f"📊 Diagnóstico API: Média de chuva prevista no estado (Próx. dias): {total_medio_futuro:.2f} mm")
    
    return chuva_df

# Mantemos a função de histórico antigo para o backtest funcionar
def coletar_dados_historicos_arquivo(data_inicio, data_fim):
    print(f"⏳ Buscando chuva histórica (Archive) de {data_inicio} a {data_fim}...")
    openmeteo = _get_openmeteo_client()
    url = "https://archive-api.open-meteo.com/v1/archive"
    lista_dfs = []
    range_datas = pd.date_range(start=data_inicio, end=data_fim, freq='D')
    
    for nome_cidade, lat, lon in CIDADES:
        params = {
            "latitude": lat, "longitude": lon,
            "start_date": data_inicio, "end_date": data_fim,
            "daily": "precipitation_sum", "timezone": "America/Sao_Paulo"
        }
        try:
            responses = openmeteo.weather_api(url, params=params)
            daily = responses[0].Daily()
            rain_values = daily.Variables(0).ValuesAsNumpy()
            
            if len(rain_values) == len(range_datas):
                df = pd.DataFrame(rain_values, index=range_datas, columns=[nome_cidade])
            else:
                df = pd.DataFrame(data=rain_values, columns=[nome_cidade])
            lista_dfs.append(df)
        except:
            lista_dfs.append(pd.DataFrame(0.0, index=range_datas, columns=[nome_cidade]))

    if not lista_dfs: return pd.DataFrame()
    return pd.concat(lista_dfs, axis=1).loc[~pd.concat(lista_dfs, axis=1).index.duplicated(keep='first')]