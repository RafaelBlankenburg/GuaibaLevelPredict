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
        # Ajuste Fino de Hiperparâmetros
        self.model = RandomForestRegressor(
            n_estimators=2000, 
            random_state=42, 
            n_jobs=-1,
            # Aumentamos leaf para 3: Isso suaviza a curva e reduz "micro-descidas" falsas
            min_samples_leaf=3, 
            max_depth=35
        )
        self.features_col_names = []
        self.is_trained = False

    def _gerar_features_df(self, df_input, cols_cidades):
        novas_colunas = {}
        
        # 1. Features Globais
        chuva_media = df_input[cols_cidades].mean(axis=1)
        novas_colunas['chuva_media_estado'] = chuva_media
        novas_colunas['saturacao_bacia_30d'] = chuva_media.rolling(window=30).sum().shift(1).fillna(0)
        
        # 2. Features por Cidade
        for col in cols_cidades:
            col_series = df_input[col]
            novas_colunas[col] = col_series
            # Lags
            for i in range(1, self.dias_atraso + 1):
                novas_colunas[f'{col}_lag_{i}'] = col_series.shift(i)
            
            # Acumulados (Foco em 3 e 7 dias)
            ac3 = col_series.rolling(window=3).sum().shift(1)
            ac7 = col_series.rolling(window=7).sum().shift(1)
            ac14 = col_series.rolling(window=14).sum().shift(1)
            
            novas_colunas[f'{col}_acum_03d'] = ac3
            novas_colunas[f'{col}_acum_07d'] = ac7
            
            # Feature de Gatilho (Chuva * Saturação)
            # Essa é a feature que permite pegar o pico da enchente
            saturacao = novas_colunas['saturacao_bacia_30d']
            novas_colunas[f'{col}_gatilho_cheia'] = ac3 * (saturacao ** 2)

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
        print("🧠 Iniciando treinamento (Foco: Picos + Estabilidade)...")
        X, y = self.preparar_dataset(df_historico)
        
        # --- ESTRATÉGIA DE PESOS DUPLA ---
        weights = np.ones(len(y))
        
        # 1. PESO DE EXPLOSÃO (Para garantir a subida no final)
        mask_subida = y > 0.02
        if mask_subida.any():
            # Aumenta exponencialmente com a altura da subida
            weights[mask_subida] = 1 + (y[mask_subida] * 60) ** 2
        
        # 2. PESO DE ESTABILIDADE (A CORREÇÃO PARA O SEU PROBLEMA)
        # Identifica dias onde o rio ficou praticamente parado (-1cm a +1cm)
        mask_estavel = (y > -0.015) & (y < 0.015)
        
        # Damos um peso muito alto (40x) para esses dias.
        # Isso ensina a IA: "Se não tem motivo claro pra mexer, FIQUE PARADO em vez de descer".
        weights[mask_estavel] = 40.0 
        
        # Clip para segurança numérica
        weights = np.clip(weights, 1, 5000)

        print(f"   Peso Máximo (Pico): {weights.max():.2f}")
        print(f"   Peso Estabilidade: 40.0")
        
        self.model.fit(X, y, sample_weight=weights)
        self.is_trained = True
        self.features_col_names = list(X.columns)
        
        print(f"✅ Treinamento concluído.")

    def salvar(self, caminho):
        if not self.is_trained: return
        os.makedirs(os.path.dirname(caminho), exist_ok=True)
        joblib.dump({'model': self.model, 'features': self.features_col_names, 'dias_atraso': self.dias_atraso}, caminho)
        print(f"💾 Modelo salvo em: {caminho}")

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

    def prever_simulacao(self, df_chuva_futura, nivel_inicial):
        if not self.is_trained: raise Exception("Modelo não treinado!")
        
        print(f"🔄 Simulando (IA Híbrida Estabilizada)...")
        
        buffer_dias = 35 
        if len(df_chuva_futura) <= buffer_dias:
            return pd.DataFrame(columns=['nivel_estimado'])

        df_simulacao = df_chuva_futura.copy()
        df_simulacao['nivel_predito'] = np.nan
        cols_cidades = [c for c in df_simulacao.columns if c != 'nivel_predito']
        resultados = []

        df_features_all = self._gerar_features_df(df_simulacao, cols_cidades)
        col_ref_3d = [c for c in df_features_all.columns if 'acum_03d' in c]

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

            # 1. Previsão da IA
            delta_ai = self.model.predict(X_input)[0]
            
            # 2. Guarda-Corpo Físico Mínimo (Apenas para garantir a subida catastrófica)
            saturacao = features_dia.get('saturacao_bacia_30d', 0)
            chuva_3d_bacia = 0
            if col_ref_3d:
                vals = [features_dia[c] for c in col_ref_3d if c in features_dia]
                if vals: chuva_3d_bacia = sum(vals) / len(vals)
            
            # Regra de Segurança: Só interfere se a situação for EXTREMA
            delta_fisico_minimo = -999
            if saturacao > 200 and chuva_3d_bacia > 50:
                delta_fisico_minimo = chuva_3d_bacia * 0.006 

            # Decisão
            delta_final = max(delta_ai, delta_fisico_minimo)
            
            # --- FILTRO DE ESTABILIDADE ---
            # Se a IA previu uma descida leve (-0.02) mas o solo não está encharcado,
            # nós zeramos a descida. Isso mantém a linha reta no início.
            if saturacao < 150: 
                if delta_final > -0.03 and delta_final < 0.01:
                    delta_final = 0.0

            novo_nivel = features_dia['nivel_anterior'] + delta_final
            
            # Limites
            if novo_nivel < 0.50: novo_nivel = 0.50
            if novo_nivel > 6.50: novo_nivel = 6.50

            df_simulacao.at[idx, 'nivel_predito'] = novo_nivel
            resultados.append({'data': idx, 'nivel_estimado': round(novo_nivel, 2)})

        return pd.DataFrame(resultados).set_index('data')