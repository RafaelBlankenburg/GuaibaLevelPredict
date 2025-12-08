import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from src.config import TARGET_COL

class PrevisorNivelRio:
    def __init__(self, dias_atraso=7):
        self.dias_atraso = dias_atraso
        self.model = RandomForestRegressor(
            n_estimators=1000, 
            random_state=42, 
            n_jobs=-1,
            min_samples_leaf=2
        )
        self.features_col_names = []
        self.is_trained = False
        self.max_saturacao_treino = 0 
        self.max_delta_treino = 0

    def _gerar_features_df(self, df_input, cols_cidades):
        novas_colunas = {}
        
        # 1. Features Globais
        chuva_media = df_input[cols_cidades].mean(axis=1)
        novas_colunas['chuva_media_estado'] = chuva_media
        novas_colunas['saturacao_bacia_15d'] = chuva_media.rolling(window=15).sum().shift(1)
        novas_colunas['saturacao_bacia_30d'] = chuva_media.rolling(window=30).sum().shift(1)
        
        # 2. Features por Cidade
        for col in cols_cidades:
            col_series = df_input[col]
            novas_colunas[col] = col_series
            for i in range(1, self.dias_atraso + 1):
                novas_colunas[f'{col}_lag_{i}'] = col_series.shift(i)
            novas_colunas[f'{col}_acum_07d'] = col_series.rolling(window=7).sum().shift(1)
            novas_colunas[f'{col}_acum_21d'] = col_series.rolling(window=21).sum().shift(1)
            saturacao = novas_colunas['saturacao_bacia_30d'].fillna(0)
            novas_colunas[f'{col}_x_saturacao'] = col_series * np.log1p(saturacao)

        return pd.DataFrame(novas_colunas, index=df_input.index)

    def preparar_dataset(self, df_historico):
        cols_cidades = [c for c in df_historico.columns if c != TARGET_COL]
        df = self._gerar_features_df(df_historico, cols_cidades)
        
        df['nivel_anterior'] = df_historico[TARGET_COL].shift(1)
        df['target_delta'] = df_historico[TARGET_COL].diff()
        df.dropna(inplace=True)
        
        self.features_col_names = [c for c in df.columns if c not in ['target_delta']]
        X = df[self.features_col_names]
        y = df['target_delta']
        
        return X, y

    def treinar(self, df_historico):
        print("🧠 Iniciando treinamento (Versão Final: Várzea Physics)...")
        X, y = self.preparar_dataset(df_historico)
        
        if 'saturacao_bacia_30d' in X.columns:
            self.max_saturacao_treino = X['saturacao_bacia_30d'].max()
        else:
            self.max_saturacao_treino = 100
            
        weights = np.abs(y)
        weights = 1 + (weights * 100)
        
        X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
            X, y, weights, test_size=0.15, shuffle=True, random_state=42
        )
        
        self.model.fit(X_train, y_train, sample_weight=w_train)
        self.is_trained = True
        self.features_col_names = list(X_train.columns)
        
        preds = self.model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        print(f"✅ Treinamento concluído. Erro Médio: {mae:.4f} m")

    def salvar(self, caminho):
        if not self.is_trained: return
        os.makedirs(os.path.dirname(caminho), exist_ok=True)
        joblib.dump({
            'model': self.model, 
            'features': self.features_col_names, 
            'dias_atraso': self.dias_atraso,
            'max_saturacao': self.max_saturacao_treino
        }, caminho)
        print(f"💾 Modelo salvo em: {caminho}")

    def carregar(self, caminho):
        if not os.path.exists(caminho): return False
        try:
            payload = joblib.load(caminho)
            self.model = payload['model']
            self.features_col_names = payload['features']
            self.dias_atraso = payload['dias_atraso']
            self.max_saturacao_treino = payload.get('max_saturacao', 200)
            self.is_trained = True
            return True
        except: return False

    def prever_simulacao(self, df_chuva_futura, nivel_inicial):
        if not self.is_trained: raise Exception("Modelo não treinado!")
        
        print(f"🔄 Simulando dia-a-dia (Com Física de Várzea)...")
        
        buffer_dias = 35 
        if len(df_chuva_futura) <= buffer_dias:
            return pd.DataFrame(columns=['nivel_estimado'])

        df_simulacao = df_chuva_futura.copy()
        df_simulacao['nivel_predito'] = np.nan
        
        cols_cidades = [c for c in df_simulacao.columns if c != 'nivel_predito']
        resultados = []

        df_features_all = self._gerar_features_df(df_simulacao, cols_cidades)

        for i in range(len(df_simulacao)):
            idx = df_simulacao.index[i]
            if i < buffer_dias: continue

            features_dia = df_features_all.iloc[i].to_dict()
            
            if i == buffer_dias:
                features_dia['nivel_anterior'] = nivel_inicial
            else:
                features_dia['nivel_anterior'] = df_simulacao.iloc[i-1]['nivel_predito']

            X_input = pd.DataFrame([features_dia])
            try:
                X_input = X_input[self.features_col_names]
            except KeyError: break

            delta_estimado = self.model.predict(X_input)[0]
            
            # --- FÍSICA HIDROLÓGICA ---
            saturacao_atual = features_dia.get('saturacao_bacia_30d', 0)
            nivel_atual = features_dia['nivel_anterior']
            
            # A. TURBO DINÂMICO (Para subir rápido no início da enchente)
            if delta_estimado > 0:
                limiar_saturacao = 150.0
                if saturacao_atual > limiar_saturacao:
                    ratio = saturacao_atual / limiar_saturacao
                    multiplicador = 1.0 + (0.3 * (ratio ** 1.8))
                    multiplicador = min(multiplicador, 3.0)
                    delta_estimado *= multiplicador
            
            # B. FREIO DE DESCIDA (Para manter o rio cheio na inércia)
            if delta_estimado < 0:
                if nivel_atual < 1.5:
                    delta_estimado *= 0.05 
                elif saturacao_atual > 50:
                    delta_estimado *= 0.2 

            # C. EFEITO VÁRZEA (NOVO: Para desacelerar no topo)
            # Acima de 3.5m, o rio espalha e perde força vertical
            if nivel_atual > 3.5 and delta_estimado > 0:
                excesso = nivel_atual - 3.5
                # Fator de amortecimento: quanto mais alto, maior o freio
                # Ex: Nível 4.5m (1m excesso) -> reduz subida em ~15%
                # Ex: Nível 5.5m (2m excesso) -> reduz subida em ~30%
                amortecimento = 1.0 - (excesso * 0.15)
                amortecimento = max(amortecimento, 0.4) # Limite mínimo
                delta_estimado *= amortecimento

            novo_nivel = nivel_atual + delta_estimado
            
            df_simulacao.at[idx, 'nivel_predito'] = novo_nivel
            resultados.append({'data': idx, 'nivel_estimado': round(novo_nivel, 2)})

        return pd.DataFrame(resultados).set_index('data')