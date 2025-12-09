import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from src.config import TARGET_COL

class PrevisorNivelRio:
    def __init__(self, dias_atraso=7):
        self.dias_atraso = dias_atraso
        self.model = RandomForestRegressor(
            n_estimators=2000, 
            random_state=42, 
            n_jobs=-1,
            min_samples_leaf=3,
            max_depth=30
        )
        self.features_col_names = []
        self.is_trained = False

    def _gerar_features_df(self, df_input, cols_cidades):
        novas_colunas = {}
        chuva_media = df_input[cols_cidades].mean(axis=1)
        novas_colunas['chuva_media_estado'] = chuva_media
        novas_colunas['saturacao_bacia_30d'] = chuva_media.rolling(window=30).sum().shift(1).fillna(0)
        
        for col in cols_cidades:
            col_series = df_input[col]
            novas_colunas[col] = col_series
            for i in range(1, self.dias_atraso + 1):
                novas_colunas[f'{col}_lag_{i}'] = col_series.shift(i)
            
            novas_colunas[f'{col}_acum_03d'] = col_series.rolling(window=3).sum().shift(1)
            novas_colunas[f'{col}_acum_07d'] = col_series.rolling(window=7).sum().shift(1)
            
            saturacao = novas_colunas['saturacao_bacia_30d']
            novas_colunas[f'{col}_x_sat'] = col_series * np.log1p(saturacao)

        return pd.DataFrame(novas_colunas, index=df_input.index)

    def preparar_dataset(self, df_historico):
        cols_cidades = [c for c in df_historico.columns if c != TARGET_COL]
        df = self._gerar_features_df(df_historico, cols_cidades)
        df['nivel_anterior'] = df_historico[TARGET_COL].shift(1)
        df['target_delta'] = df_historico[TARGET_COL].diff()
        df.dropna(inplace=True)
        self.features_col_names = [c for c in df.columns if c not in ['target_delta']]
        return df[self.features_col_names], df['target_delta']

    def treinar(self, df_historico):
        print("🧠 Iniciando treinamento (Pesos Híbridos 2023/24)...")
        X, y = self.preparar_dataset(df_historico)
        
        weights = np.ones(len(y))
        
        # Pesos: Mantemos o equilíbrio que funcionou
        mask_subida = y > 0.02
        if mask_subida.any():
            weights[mask_subida] = 1 + (y[mask_subida] * 150)
        
        mask_estavel = (y > -0.015) & (y < 0.015)
        weights[mask_estavel] = 30.0 
        
        weights = np.clip(weights, 1, 5000)

        self.model.fit(X, y, sample_weight=weights)
        self.is_trained = True
        self.features_col_names = list(X.columns)
        print(f"✅ Treinamento concluído.")

    def salvar(self, caminho):
        if not self.is_trained: return
        os.makedirs(os.path.dirname(caminho), exist_ok=True)
        joblib.dump({'model': self.model, 'features': self.features_col_names, 'dias_atraso': self.dias_atraso}, caminho)

    def carregar(self, caminho):
        if not os.path.exists(caminho): return False
        try:
            payload = joblib.load(caminho)
            self.model = payload['model']
            self.features_col_names = payload['features']
            self.dias_atraso = payload['dias_atraso']
            self.is_trained = True
            return True
        except: return False

    def prever_simulacao(self, df_chuva_futura, nivel_inicial, data_inicio_simulacao=None):
        if not self.is_trained: raise Exception("Modelo não treinado!")
        print(f"🔄 Simulando (Com Drenagem Inteligente)...")
        
        df_simulacao = df_chuva_futura.copy()
        df_simulacao['nivel_predito'] = np.nan
        cols_cidades = [c for c in df_simulacao.columns if c != 'nivel_predito']
        resultados = []

        df_features_all = self._gerar_features_df(df_simulacao, cols_cidades)
        col_ref_3d = [c for c in df_features_all.columns if 'acum_03d' in c]

        ultimo_nivel_conhecido = nivel_inicial
        simulacao_ativa = False
        data_alvo = pd.to_datetime(data_inicio_simulacao) if data_inicio_simulacao else None
        buffer_idx = 35

        for i in range(len(df_simulacao)):
            idx = df_simulacao.index[i]
            if i < buffer_idx: continue

            if data_alvo:
                if idx < data_alvo: continue
                elif idx == data_alvo:
                    ultimo_nivel_conhecido = nivel_inicial
                    simulacao_ativa = True
                    resultados.append({'data': idx, 'nivel_estimado': nivel_inicial})
                    continue
            else:
                simulacao_ativa = True

            if not simulacao_ativa: continue

            features_dia = df_features_all.iloc[i].to_dict()
            features_dia['nivel_anterior'] = ultimo_nivel_conhecido

            X_input = pd.DataFrame([features_dia])
            try:
                X_input = X_input[self.features_col_names]
            except KeyError: break

            delta_ai = self.model.predict(X_input)[0]
            
            # --- FÍSICA AVANÇADA: ENTRADA vs SAÍDA ---
            
            saturacao = features_dia.get('saturacao_bacia_30d', 0)
            chuva_3d_bacia = 0
            if col_ref_3d:
                vals = [features_dia[c] for c in col_ref_3d if c in features_dia]
                if vals: chuva_3d_bacia = sum(vals) / len(vals)
            
            # 1. ENTRADA (Runoff) - Ajustado para 2023
            delta_fisico_minimo = -999
            if chuva_3d_bacia > 50:
                if saturacao > 300: # Catástrofe (2024)
                    delta_fisico_minimo = chuva_3d_bacia * 0.006 
                elif saturacao > 180: # Cheia Média (2023)
                    delta_fisico_minimo = chuva_3d_bacia * 0.0025 # Reduzi um pouco mais (era 0.0035)

            # 2. SAÍDA (Drenagem Variável)
            # Quanto maior o nível, maior a pressão para a água sair.
            # Mas se a saturação for extrema (2024), a saída está "entupida".
            
            fator_drenagem = 0.0
            if ultimo_nivel_conhecido > 1.50:
                if saturacao > 300:
                    # 2024: Drenagem lenta (represamento)
                    fator_drenagem = 0.03
                else:
                    # 2023: Drenagem rápida (escoamento livre)
                    # O rio perde força conforme sobe
                    fator_drenagem = 0.06
            
            drenagem_natural = 0.0
            if ultimo_nivel_conhecido > 1.20:
                drenagem_natural = (ultimo_nivel_conhecido - 1.20) * fator_drenagem

            # Decisão Final
            # A física impulsiona pra cima, a drenagem puxa pra baixo
            delta_final = max(delta_ai, delta_fisico_minimo)
            
            # Aplica a drenagem calculada
            delta_final -= drenagem_natural

            # Filtro de Estabilidade (Chão)
            if saturacao < 150: 
                if delta_final > -0.03 and delta_final < 0.01:
                    delta_final = 0.0

            novo_nivel = ultimo_nivel_conhecido + delta_final
            
            if novo_nivel < 0.50: novo_nivel = 0.50
            if novo_nivel > 6.00: novo_nivel = 6.00

            ultimo_nivel_conhecido = novo_nivel
            df_simulacao.at[idx, 'nivel_predito'] = novo_nivel
            resultados.append({'data': idx, 'nivel_estimado': round(novo_nivel, 2)})

        return pd.DataFrame(resultados).set_index('data')