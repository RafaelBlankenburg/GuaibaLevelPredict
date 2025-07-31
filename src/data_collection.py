import pandas as pd
import openmeteo_requests
import requests_cache
from retry_requests import retry


CIDADES = [
    ("vacaria_mm", -28.5122, -50.9339), ("guapore_mm", -28.8456, -51.8903),
    ("lagoa_vermelha_mm",-28.2086,-51.5258), ("passo_fundo_mm",-28.2628,-52.4067),
    ("soledade_mm", -28.8183, -52.5103), ("cruz_alta_mm",-28.644,-53.6063),
    ("salto_do_jacui_mm",-29.0883,-53.2125), ("sao_francisco_de_paula_mm",-29.4481,-50.5836),
    ("bento_goncalves_mm",-29.1714,-51.5192), ("caxias_do_sul_mm",-29.1681,-51.1794),
    ("lajeado_mm",-29.4669,-51.9614), ("taquari_mm", -29.7997, -51.8644),
    ("santa_cruz_do_sul_mm",-29.7175,-52.4258), ("julio_de_castilhos_mm",-29.2269,-53.6817),
    ("santa_maria_mm", -29.6842, -53.8069), ("viamao_mm",-30.0811,-51.0233),
    ("cachoeira_do_sul_mm",-30.0392,-52.8939), ("encruzilhada_do_sul_mm",-30.5439,-52.5219),
    ("cacapava_do_sul_mm",-30.5144,-53.485), ("sao_gabriel_mm",-30.3364,-54.32)
]

def _get_openmeteo_client():
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    return openmeteo_requests.Client(session=retry(cache_session, retries=5, backoff_factor=0.2))

def coletar_dados_chuva(dias_historico, dias_previsao):
    print(f"⏳ Coletando dados de chuva ({dias_historico} dias de histórico + {dias_previsao} dias de previsão)...")
    openmeteo = _get_openmeteo_client()
    dfs = []
    all_dates = None # Inicializa a variável de datas como nula

    for nome_cidade, lat, lon in CIDADES:
        params = {
            "latitude": lat, "longitude": lon, "daily": "rain_sum",
            "timezone": "America/Sao_Paulo", "past_days": dias_historico, "forecast_days": dias_previsao
        }
        try:
            responses = openmeteo.weather_api("https://api.open-meteo.com/v1/forecast", params=params)
            response = responses[0]
            daily = response.Daily()

            if daily and daily.Variables(0):
                rain_values = daily.Variables(0).ValuesAsNumpy()
                df = pd.DataFrame(data=rain_values, columns=[nome_cidade])

                if all_dates is None:
                    start_date_unix = daily.Time()
                    start_date = pd.to_datetime(start_date_unix, unit="s").date()
                    all_dates = pd.date_range(start=start_date, periods=len(df), freq='D')
            else:
                num_dias_total = (dias_historico or 0) + (dias_previsao or 0)
                df = pd.DataFrame(np.zeros(num_dias_total), columns=[nome_cidade])
                print(f"⚠️  Aviso: Não foram retornados dados para {nome_cidade}.")

            dfs.append(df)

        except Exception as e:
            print(f"❌ Erro na API para {nome_cidade}: {e}. Preenchendo com zeros.")
            num_dias_total = (dias_historico or 0) + (dias_previsao or 0)
            dfs.append(pd.DataFrame(np.zeros(num_dias_total), columns=[nome_cidade]))
    
    chuva_df = pd.concat(dfs, axis=1)
    
    if all_dates is None and len(chuva_df) > 0:
        print("❌ ERRO GERAL DE API: Não foi possível determinar o período. Gerando datas de fallback.")
        hoje = pd.to_datetime('today').normalize().date()
        data_inicio_fallback = hoje - pd.Timedelta(days=dias_historico)
        all_dates = pd.date_range(start=data_inicio_fallback, periods=len(chuva_df), freq='D')

    if all_dates is not None:
        chuva_df = chuva_df.head(len(all_dates))
        chuva_df.index = all_dates
    
    # --- DIAGNÓSTICO ADICIONADO NOVAMENTE ---
    print("\n--- DADOS DE CHUVA COLETADOS (HISTÓRICO + PREVISÃO) ---")
    print(chuva_df)
    print("------------------------------------------------------\n")
    
    print("✅ Dados de chuva coletados.")
    return chuva_df
    
def coletar_nivel_atual_rio():
    print("⚠️  Atenção: Usando valor fixo para o nível atual do rio. Implementar busca real.")
    return 1.22 # Exemplo: 2.34 metros
    