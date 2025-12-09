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
            min_samples_leaf=2,
            max_depth=20 # Reduzi a profundidade para evitar overfitting em picos falsos
        )
        self.features_col_names = []
        self.is_trained = False

    def _gerar_features_df(self, df_input, cols_cidades):
        novas_colunas = {}
        
        # 1. Features Globais
        chuva_media = df_input[cols_cidades].mean(axis=1)
        novas_colunas['chuva_media_estado'] = chuva_media
        
        # Saturação
        saturacao_30d = chuva_media.rolling(window=30).sum().shift(1).fillna(0)
        novas_colunas['saturacao_bacia_30d'] = saturacao_30d
        
        # 2. Features por Cidade
        for col in cols_cidades:
            col_series = df_input[col]
            novas_colunas[col] = col_series
            # Lags
            for i in range(1, self.dias_atraso + 1):
                novas_colunas[f'{col}_lag_{i}'] = col_series.shift(i)
            # Acumulados
            novas_colunas[f'{col}_acum_07d'] = col_series.rolling(window=7).sum().shift(1)
            novas_colunas[f'{col}_acum_14d'] = col_series.rolling(window=14).sum().shift(1)
            
            # Feature de Interação
            novas_colunas[f'{col}_x_sat'] = col_series * np.log1p(saturacao_30d)

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
        print("🧠 Iniciando treinamento (Calibragem Suave)...")
        X, y = self.preparar_dataset(df_historico)
        
        # --- CALIBRAGEM 1: Menos Clonagem ---
        # Antes: 20x | Agora: 5x
        # O modelo precisa ver enchentes, mas não pode achar que todo dia é enchente
        mask_subida = y > 0.05
        X_subidas = X[mask_subida]
        y_subidas = y[mask_subida]
        
        if len(X_subidas) > 0:
            X_final = pd.concat([X] + [X_subidas] * 5)
            y_final = pd.concat([y] + [y_subidas] * 5)
        else:
            X_final, y_final = X, y
        
        # --- CALIBRAGEM 2: Pesos Menores ---
        # Antes: 10.0 | Agora: 3.0
        weights = np.ones(len(y_final))
        weights[y_final.values > 0] = 3.0 
        
        self.model.fit(X_final, y_final, sample_weight=weights)
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
        
        print(f"🔄 Simulando (Modo: Estável)...")
        
        buffer_dias = 35 
        if len(df_chuva_futura) <= buffer_dias:
            return pd.DataFrame(columns=['nivel_estimado'])

        df_simulacao = df_chuva_futura.copy()
        df_simulacao['nivel_predito'] = np.nan
        cols_cidades = [c for c in df_simulacao.columns if c != 'nivel_predito']
        resultados = []

        df_features_all = self._gerar_features_df(df_simulacao, cols_cidades)

        col_ref = [c for c in df_features_all.columns if 'acum_07d' in c]

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

            # 1. IA Pura
            delta_ai = self.model.predict(X_input)[0]
            
            # 2. CÁLCULO FÍSICO (CALIBRADO)
            chuva_acum_bacia = 0
            if col_ref:
                vals = [features_dia[c] for c in col_ref if c in features_dia]
                if vals: chuva_acum_bacia = sum(vals) / len(vals)
            
            saturacao = features_dia.get('saturacao_bacia_30d', 0)
            
            # --- CALIBRAGEM 3: Coeficientes mais mansos ---
            coeficiente = 0.001 
            if saturacao > 100: coeficiente = 0.002
            if saturacao > 200: coeficiente = 0.004 
            if saturacao > 300: coeficiente = 0.006 # Cortei pela metade (era 0.012)

            delta_fisico = (chuva_acum_bacia * coeficiente) / 2.0 
            
            # --- DECISÃO HÍBRIDA ---
            if chuva_acum_bacia > 20:
                delta_final = max(delta_ai, delta_fisico)
                
                # Removemos o "Turbo 1.5x" aqui para evitar explosão.
                # Apenas confiamos na física pura calibrada.
            else:
                delta_final = delta_ai
                
                # Freio de descida
                nivel_atual = features_dia['nivel_anterior']
                if delta_final < 0:
                    if nivel_atual < 1.0: delta_final *= 0.05
                    elif saturacao > 60: delta_final *= 0.2

            # --- AMORTECIMENTO DE TOPO (EFEITO VÁRZEA) ---
            # Se o rio já está muito alto (>4.5m), ele desacelera a subida
            nivel_atual = features_dia['nivel_anterior']
            if nivel_atual > 4.5 and delta_final > 0:
                # Reduz a velocidade em 40%
                delta_final *= 0.6

            novo_nivel = nivel_atual + delta_final
            
            # Travas de segurança absolutas
            if novo_nivel < 0.50: novo_nivel = 0.50
            if novo_nivel > 6.50: novo_nivel = 6.50 # Impede o gráfico de 140m

            df_simulacao.at[idx, 'nivel_predito'] = novo_nivel
            resultados.append({'data': idx, 'nivel_estimado': round(novo_nivel, 2)})

        return pd.DataFrame(resultados).set_index('data')