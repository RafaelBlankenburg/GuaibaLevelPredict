from src.train import train_model
from src.collect_rain_data import coletar_chuva_14dias, prever_nivel_rio

if __name__ == "__main__":
    print("⚙️  Iniciando treinamento do modelo com dados históricos...")
    train_model()
    print("✅ Treinamento concluído e modelo salvo em /models")

    print("\n⏳ Coletando previsões de chuva para os próximos 14 dias via Open-Meteo...")
    df_chuva = coletar_chuva_14dias()
    df_chuva.to_csv("data/chuvas_14dias.csv", index=False)
    print("✅ Dados salvos em data/chuvas_14dias.csv")

    print("\n🔮 Gerando previsões do nível do rio com modelo LSTM...")
    previsoes = prever_nivel_rio(df_chuva)
    previsoes.to_csv("data/previsao_nivel_rio.csv", index=False)

    print("\n📈 Previsões do nível do rio:\n")
    print(previsoes)
