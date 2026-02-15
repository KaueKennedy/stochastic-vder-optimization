import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os
import numpy as np
from scipy.interpolate import griddata
import requests
import json
import os
import streamlit as st
import base64

# --- CONFIGURATION (Caminhos Limpos) ---
# Usa a mesma lógica do config.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Pasta code
ROOT_DIR = os.path.dirname(BASE_DIR)                  # Pasta codes
RESULTS_DIR = os.path.join(ROOT_DIR, "resultados")    # Pasta resultados

# Arquivos
DATA_FILE = os.path.join(RESULTS_DIR, "RELATORIO_COMPLETO_SCORES.csv")
BUS_FILE = os.path.join(RESULTS_DIR, "MASTER_Bus_Results.csv")
VOLT_FILE = os.path.join(RESULTS_DIR, "MASTER_Voltage_Log.csv")
HIST_FILE = os.path.join(RESULTS_DIR, "Optimization_History_Log.csv")
HOURLY_FILE = os.path.join(RESULTS_DIR, "MASTER_Hourly_Results.csv")

EXCEL_REDE = os.path.join(ROOT_DIR, "Iowa_Distribution_Test_Systems", "OpenDSS Model", "OpenDSS Model", "Rede_240Bus_Dados.xlsx")

# Dicionário para Contexto da IA
CONTEXT_FILES = {
    "Hourly": HOURLY_FILE,
    "Bus": BUS_FILE,
    "Voltage": VOLT_FILE,
    "History": HIST_FILE
}

# ==============================================================================
# METADADOS DOS INDICADORES (19 Definições cobrem as 35 Colunas)
# ==============================================================================
METRIC_INFO = {
    # --- 1. GLOBAL ---
    'GLOBAL_PERFORMANCE_INDEX': {
        'en': 'Global Performance Index. Aggregate score (0-100) averaging all optimized indicators.',
        'fr': 'Indice de Performance Global. Score agrégé (0-100) moyennant tous les indicateurs optimisés.',
        'eq': r''' Index = \frac{1}{N} \sum_{i=1}^{N} Score_i '''
    },

    # --- 2. DADOS ABSOLUTOS (Apenas Brutos) ---
    'Total_Capex': {
        'en': 'Total CAPEX ($). Total initial capital expenditure required for the DER configuration.',
        'fr': 'CAPEX Total ($). Dépenses d\'investissement initiales totales requises.',
        'eq': r''' CAPEX = \sum (Cost_{kW} \times Capacity_{kW}) '''
    },
    'Total_Revenue_Daily': {
        'en': 'Daily Revenue ($). Total daily income from Energy sales, Grid Services, and Incentives.',
        'fr': 'Revenu Quotidien ($). Revenu total journalier (Vente d\'énergie, Services réseau, Incitations).',
        'eq': r''' Rev_{day} = \sum_{t=0}^{24} (P_{inj} \times Price_t) + Incentives '''
    },

    # --- 3. FINANCEIROS (Bruto + Score) ---
    'ROI_10y_Pct': {
        'en': '10-Year ROI (%). Return on Investment over 10 years, considering CAPEX and daily revenue.',
        'fr': 'ROI sur 10 ans (%). Retour sur investissement sur 10 ans, considérant CAPEX et revenus.',
        'eq': r''' ROI = \frac{(Rev_{day} \times 3650) - CAPEX}{CAPEX} \times 100 '''
    },
    'LCOE_USD_kWh': {
        'en': 'LCOE ($/kWh). Levelized Cost of Energy. Average cost per kWh generated over system life.',
        'fr': 'LCOE ($/kWh). Coût actualisé de l\'énergie. Coût moyen par kWh généré sur la vie du système.',
        'eq': r''' LCOE = \frac{CAPEX + \sum O\&M}{(Gen_{MWh/yr} \times 20yr) \times 1000} '''
    },
    'Windfall_Profit_Index': {
        'en': 'Windfall Profit Index. Measures excess returns above the regulated Cost of Capital (WACC).',
        'fr': 'Indice de Profit Exceptionnel. Mesure les rendements excédentaires au-dessus du WACC.',
        'eq': r''' Windfall = \frac{ROI_{\%} - WACC_{\%}}{WACC_{\%}} '''
    },
    'Revenue_Volatility': {
        'en': 'Revenue Volatility (%). Coefficient of variation of the hourly revenue stream (Risk metric).',
        'fr': 'Volatilité des Revenus (%). Coefficient de variation du flux de revenus horaire (Risque).',
        'eq': r''' Vol = \frac{\sigma_{RevHourly}}{\mu_{RevHourly}} \times 100 '''
    },

    # --- 4. TÉCNICOS (Bruto + Score) ---
    'Self_Sufficiency_Pct': {
        'en': 'Self-Sufficiency (%). Percentage of total load demand met by local generation.',
        'fr': 'Auto-suffisance (%). Pourcentage de la demande totale couverte par la production locale.',
        'eq': r''' SS = \frac{Gen_{Total}}{Load_{Total}} \times 100 '''
    },
    'Peak_Coincidence_Pct': {
        'en': 'Peak Coincidence (%). Share of DG injection occurring during grid peak hours.',
        'fr': 'Coïncidence de Pointe (%). Part de l\'injection GD pendant les heures de pointe.',
        'eq': r''' PC = \frac{\sum_{t \in Peak} Gen_t}{\sum_{t \in Peak} Load_t} \times 100 '''
    },
    'Voltage_Deviation_Avg': {
        'en': 'Avg Voltage Deviation (pu). Average absolute difference from 1.0 p.u. across all buses.',
        'fr': 'Déviation de Tension Moy (pu). Différence absolue moyenne par rapport à 1.0 p.u.',
        'eq': r''' V_{dev} = \frac{1}{N} \sum_{i=1}^{N} |V_i - 1.0| '''
    },
    'Grid_Congestion_Max_Pct': {
        'en': 'Max Congestion (%). Highest loading percentage found on lines or transformers.',
        'fr': 'Congestion Max (%). Charge maximale détectée sur les lignes ou transformateurs.',
        'eq': r''' Cong = \max \left( \frac{I_{flow}}{I_{limit}} \right) \times 100 '''
    },
    'Tech_Losses_Total_kWh': {
        'en': 'Technical Losses (kWh). Total energy dissipated due to resistance (Joule effect).',
        'fr': 'Pertes Techniques (kWh). Énergie totale dissipée par effet Joule.',
        'eq': r''' Loss = \sum_{t} \sum_{lines} 3 \cdot R \cdot I(t)^2 '''
    },

    # --- 5. SÓCIO-AMBIENTAL & POLÍTICA (Bruto + Score) ---
    'CO2_Avoided_Tons_Year': {
        'en': 'CO2 Avoided (Metric Tons/yr). Based on grid intensity of 632 lbs/MWh.',
        'fr': 'CO2 Évité (Tonnes/an). Basé sur une intensité réseau de 632 lbs/MWh.',
        'eq': r''' CO2 = Gen_{MWh} \times \frac{632 \text{ lbs}}{2204.6} \approx Gen \times 0.287 \, t/MWh '''
    },
    'Social_Equity_Pct': {
        'en': 'Social Equity (%). Percentage of total capacity installed in Rural zones.',
        'fr': 'Équité Sociale (%). Pourcentage de la capacité totale installée en zone rurale.',
        'eq': r''' Eq = \frac{Cap_{Rural}}{Cap_{Total}} \times 100 '''
    },
    'Gini_Incentivo': {
        'en': 'Incentive Gini Coeff. Measures inequality in capacity distribution (0=Equal, 1=Concentrated).',
        'fr': 'Coeff. Gini Incitatif. Mesure l\'inégalité de distribution (0=Égal, 1=Concentré).',
        'eq': r''' G = \frac{\sum (2i - n - 1) x_i}{n \sum x_i} '''
    },
    'Revenue_Disparity_Ratio': {
        'en': 'Rev. Disparity Ratio. Ratio of Rural Revenue/kW vs Urban Revenue/kW (Target > 1.0).',
        'fr': 'Ratio de Disparité. Rapport Revenu/kW Rural vs Revenu/kW Urbain (Cible > 1.0).',
        'eq': r''' Ratio = \frac{Rev_{Rural}/kW}{Rev_{Urb}/kW} '''
    },
    'Subsidy_Intensity': {
        'en': 'Subsidy Intensity ($/MWh). Public cost per unit of renewable energy generated.',
        'fr': 'Intensité des Subventions. Coût public par unité d\'énergie générée.',
        'eq': r''' Int = \frac{TotalSubsidy}{TotalGen_{MWh}} '''
    },
    'Private_Investment_Leverage': {
        'en': 'Private Leverage. Ratio of Private Investment (CAPEX) to Public Subsidy (10y).',
        'fr': 'Levier Privé. Ratio Investissement Privé (CAPEX) / Subvention Publique (10 ans).',
        'eq': r''' Lev = \frac{CAPEX}{Subsidy_{yr} \times 10} '''
    },
    'Incentivized_Carbon_Cost': {
        'en': 'Incentivized Carbon Cost ($/tCO2). Public subsidy spent per ton of CO2 avoided.',
        'fr': 'Coût Carbone Incitatif. Subvention publique par tonne de CO2 évitée.',
        'eq': r''' Cost = \frac{Subsidy_{EnvPart}}{CO2_{Avoided}} '''
    }
}

DEFAULT_INFO = {
    'en': 'Normalized score or raw value.',
    'fr': 'Score normalisé ou valeur brute.',
    'eq': r''' Score = \frac{Value - Min}{Max - Min} \times 100 '''
}

# ==============================================================================
# AI ENGINE (OLLAMA)
# ==============================================================================
def query_ollama(prompt, model="llama3.2"):
    url = "http://localhost:11434/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            return response.json().get('response', "No response.")
        return f"Error {response.status_code}"
    except:
        return "Connection Error: Is Ollama running?"

def build_ai_prompt(df_slice, x_axis, y_axis, z_axis, fixed_filters):
    stats = df_slice[z_axis].describe().to_string()
    
    # Contexto dos arquivos existentes
    files_status = []
    for k, v in CONTEXT_FILES.items():
        if os.path.exists(v): files_status.append(k)
        
    prompt = f"""
    You are an expert in Distributed Energy Resources (DER) Analysis.
    
    GRAPH CONTEXT:
    - X-Axis (Input Weight): {x_axis}
    - Y-Axis (Input Weight): {y_axis}
    - Z-Axis (Result): {z_axis}
    - Fixed Variables: {fixed_filters}
    - Available Datasets: {', '.join(files_status)}
    
    STATISTICS (Z-Axis):
    {stats}
    
    TASK:
    Analyze the relationship shown in this graph. 
    1. Does increasing {x_axis} or {y_axis} lead to better {z_axis}?
    2. Are there diminishing returns or trade-offs?
    3. Provide a concise, professional insight (max 100 words).
    """
    return prompt

# ==============================================================================
# APP MAIN
# ==============================================================================
@st.cache_data
def load_data():
    """
    Carrega o arquivo principal e cruza com os resultados de barra (Bus Results)
    para calcular métricas zonais (Rural/Urbano) sem precisar rodar a simulação de novo.
    """
    # 1. Carrega Scores Principais
    if not os.path.exists(DATA_FILE):
        st.error(f"❌ Arquivo principal ausente: {DATA_FILE}")
        return None
    df_main = pd.read_csv(DATA_FILE)

    # 2. Se já tiver as colunas, retorna direto
    if 'Rev_per_kW_Rural' in df_main.columns:
        return df_main.round(4)

    # 3. Se faltar, tenta calcular via Engenharia Reversa
    st.sidebar.caption("🔄")
    
    if os.path.exists(BUS_FILE) and os.path.exists(EXCEL_REDE):
        try:
            # A. Carrega Resultados detalhados
            df_bus = pd.read_csv(BUS_FILE)
            
            # B. Carrega Mapa do Excel
            # (Normaliza nomes para minúsculo para garantir o 'match')
            df_excel = pd.read_excel(EXCEL_REDE, sheet_name='Buses')
            zone_map = dict(zip(
                df_excel['BusName'].astype(str).str.strip().str.lower(), 
                df_excel['Zone'].astype(str).str.strip().str.title()
            ))
            
            # C. Aplica o Mapa
            df_bus['Zone'] = df_bus['Bus'].astype(str).str.strip().str.lower().map(lambda x: zone_map.get(x, 'Mixed'))
            
            # D. Pivot para somar por Sim_ID e Zona
            # Cria colunas: Cap_kW_Rural, Gen_kWh_Rural, etc.
            p_cap = df_bus.pivot_table(index='Sim_ID', columns='Zone', values='Capacity (kW)', aggfunc='sum').fillna(0).add_prefix('Cap_kW_')
            p_gen = df_bus.pivot_table(index='Sim_ID', columns='Zone', values='Total Gen (kWh)', aggfunc='sum').fillna(0).add_prefix('Gen_kWh_')
            
            # E. Merge com DataFrame Principal (Garante que Sim_ID seja string)
            df_main['Sim_ID'] = df_main['Sim_ID'].astype(str)
            p_cap.index = p_cap.index.astype(str)
            p_gen.index = p_gen.index.astype(str)
            
            df_merged = df_main.merge(p_cap, on='Sim_ID', how='left').merge(p_gen, on='Sim_ID', how='left')
            
            # F. Calcula Métricas Finais (Rev/kW e %)
            base_price = 0.10 # Preço base referência
            total_cap = df_merged[['Cap_kW_Rural', 'Cap_kW_Mixed', 'Cap_kW_Urban']].sum(axis=1).replace(0, 1)

            for z in ['Rural', 'Mixed', 'Urban']:
                # Garante que colunas existam
                if f'Cap_kW_{z}' not in df_merged: df_merged[f'Cap_kW_{z}'] = 0
                if f'Gen_kWh_{z}' not in df_merged: df_merged[f'Gen_kWh_{z}'] = 0
                
                # % Capacidade
                df_merged[f'Cap_Pct_{z}'] = (df_merged[f'Cap_kW_{z}'] / total_cap * 100)
                
                # Receita Estimada ($/kW)
                # Rural ganha bônus W_Social. Urbano não.
                bonus = df_merged['W_Social'] if z == 'Rural' else 0.0
                rev = df_merged[f'Gen_kWh_{z}'] * base_price * (1 + bonus)
                cap = df_merged[f'Cap_kW_{z}'].replace(0, np.nan)
                df_merged[f'Rev_per_kW_{z}'] = (rev / cap).fillna(0)

            st.sidebar.success("✅")
            return df_merged.round(4)

        except Exception as e:
            st.sidebar.error(f"Erro processando barras: {e}")
            return df_main.round(4)
    else:
        st.sidebar.warning("⚠️ Arquivos auxiliares (Bus/Excel) não encontrados.")
        return df_main.round(4)

def main():
    st.set_page_config(layout="wide", page_title="Analyse des DERs", page_icon="🏔️")
    st.title("Analyse des Subventions et Performances des DER")
    
    df = load_data()
    if df is None:
        st.error(f"Data missing: {DATA_FILE}")
        st.stop()

    cols_factors = ['W_Grid', 'W_Cap', 'W_Social', 'W_Env']
    cols_scores = [c for c in df.columns if 'Score_' in c] + ['GLOBAL_PERFORMANCE_INDEX']
    ignore = cols_factors + cols_scores + ['Sim_ID', 'Timestamp']
    cols_raw = [c for c in df.columns if c not in ignore and df[c].dtype in ['float64', 'int64']]
    all_metrics = sorted(cols_scores + cols_raw)

    # --- SIDEBAR ---
    st.sidebar.header("1. Axis Setup")
    
    try: idx_x = cols_factors.index('W_Social')
    except: idx_x = 0
    axis_x = st.sidebar.selectbox("X-Axis (Input)", cols_factors, index=idx_x)
    
    remaining = [c for c in cols_factors if c != axis_x]
    try: idx_y = remaining.index('W_Env')
    except: idx_y = 0
    axis_y = st.sidebar.selectbox("Y-Axis (Input)", remaining, index=idx_y)
    
    st.sidebar.header("2. Filters")
    filter_vars = [c for c in cols_factors if c not in [axis_x, axis_y]]
    current_filters = {}
    mask = pd.Series([True] * len(df))  # Começa com TUDO

    for f in filter_vars:
        available_vals = sorted(df[f].dropna().unique())
        if len(available_vals) < 2:  # Skip se só 1 valor
            continue
        
        # Default: MÉDIA (não meio da lista)
        default_val = df[f].median()
        val = st.sidebar.select_slider(
            f"Fix {f}:", 
            options=available_vals, 
            value=default_val  # ← MELHOR DEFAULT
        )
        current_filters[f] = val
        mask &= (df[f] == val)

    df_3d = df[mask].copy()

    # ← CRÍTICO: DEBUG e Proteção
    st.sidebar.markdown("---")
    st.sidebar.metric("📊 Filtered Rows", len(df_3d))
    if len(df_3d) == 0:
        st.sidebar.error("❌ **NO DATA** after filters! Reset filters.")
        df_3d = df.copy()  # ← FORÇA fallback para TUDO
        st.sidebar.warning("🔄 **Auto-reset**: Showing ALL data.")

    st.sidebar.caption(f"Filters: {current_filters}")


    # --- TABS ---
    tab_3d, tab_corr, tab_zones = st.tabs(["🏔️ 3D Landscape Analysis", "🔥 Correlations", "🌍 Zonal Analysis"])


    with tab_3d:
        if df_3d.empty:
            st.warning("No data. Try looser filters.")
            st.stop()
        # Z Selector
        c_sel, _ = st.columns([1, 3])
        with c_sel:
            def_ix = all_metrics.index('GLOBAL_PERFORMANCE_INDEX') if 'GLOBAL_PERFORMANCE_INDEX' in all_metrics else 0
            axis_z = st.selectbox("Select Indicator (Z-Axis):", all_metrics, index=def_ix)

            st.sidebar.markdown("---")
            if st.sidebar.button("💾 **EXPORT ALL GRAPHS → XLS**"):
                export_file = export_all_to_xls(df, df_3d, axis_x, axis_y, axis_z, current_filters, cols_factors)

        # 3D Plot
        if not df_3d.empty:
            try:
                # 1. Interpolation Grid
                xi = np.linspace(df_3d[axis_x].min(), df_3d[axis_x].max(), 50)
                yi = np.linspace(df_3d[axis_y].min(), df_3d[axis_y].max(), 50)
                Xi, Yi = np.meshgrid(xi, yi)
                Zi = griddata((df_3d[axis_x], df_3d[axis_y]), df_3d[axis_z], (Xi, Yi), method='linear')

                # Check min/max z values
                z_min = np.nanmin(Zi)
                z_max = np.nanmax(Zi)

                # Base settings for Surface
                surface_kwargs = {
                    'z': Zi, 'x': Xi, 'y': Yi,
                    'colorscale': 'Viridis', # Azul (baixo) -> Amarelo (alto)
                    'opacity': 0.95,
                    'colorbar': dict(title=axis_z)
                }
                # Base settings for Z-Axis Layout
                layout_z_settings = dict(title=axis_z)

                # --- CORREÇÃO PARA "MESA PLANA" (FLAT SURFACE) ---
                if np.isclose(z_min, z_max):
                    val = z_min
                    st.info(f"ℹ️ Constant value detected: {val:.4f}. Showing flat surface.")
                    
                    if np.isclose(val, 0):
                        # CASO ZERO: Força a cor Azul (início da escala Viridis)
                        # Definimos que a escala de cor vai de 0.0 a 1.0. Como o valor é 0.0, ele pega a primeira cor.
                        surface_kwargs['cmin'] = 0.0
                        surface_kwargs['cmax'] = 1.0 # Valor arbitrário maior que 0 para criar gradiente
                        # Força o eixo Z a ter uma altura visual pequena ao redor de 0
                        layout_z_settings['range'] = [-0.1, 0.1]
                    else:
                        # CASO CONSTANTE NÃO-ZERO (ex: 567.29): Centraliza a cor (fica esverdeado)
                        margin = abs(val) * 0.01 # Margem de 1%
                        surface_kwargs['cmin'] = val - margin
                        surface_kwargs['cmax'] = val + margin
                        layout_z_settings['range'] = [val - margin, val + margin]

                # 2. Create Figure with dynamic settings
                fig = go.Figure(data=[go.Surface(**surface_kwargs)])
                
                fig.update_layout(
                    title=f"Terrain: {axis_z}",
                    scene=dict(
                        xaxis=dict(title=axis_x),
                        yaxis=dict(title=axis_y),
                        zaxis=layout_z_settings, # Aplica o range corrigido
                        aspectmode='cube'
                    ),
                    height=600, margin=dict(l=0, r=0, b=0, t=30)
                )
                st.plotly_chart(fig, width="stretch")

            except Exception as e:
                st.error(f"Error plotting 3D surface (possibly insufficient distinct points): {e}")
                # Fallback opcional: mostrar pontos se a superfície falhar
                st.plotly_chart(px.scatter_3d(df_3d, x=axis_x, y=axis_y, z=axis_z), width="stretch")
        else:
            st.warning("No data found for these filters.")

        # INFO & AI Columns
        st.markdown("---")
        c_def, c_ai = st.columns([1, 1])

        with c_def:
            st.subheader("📖 Indicator Definition")
            
            # --- LÓGICA DE VÍNCULO INTELIGENTE ---
            lookup_key = axis_z
            is_score = False
            
            if axis_z.startswith("Score_"):
                lookup_key = axis_z.replace("Score_", "")
                is_score = True
            
            info = METRIC_INFO.get(lookup_key, DEFAULT_INFO)
            
            # Exibição
            prefix = "**(Normalized Score 0-100)** of: " if is_score else ""
            st.markdown(f"🇬🇧 **En:** {prefix}{info.get('en', '')}")
            st.markdown(f"🇫🇷 **Fr:** {prefix}{info.get('fr', '')}")
            
            st.markdown("#### Formula")
            if is_score:
                st.latex(r''' Score = \frac{Value - Min}{Max - Min} \times 100 ''')
                st.caption("Underlying metric calculation:")
            st.latex(info.get('eq', ''))

        with c_ai:
            st.subheader("🤖 AI Analyst")
            
            # 1. Análise Padrão
            if st.button("🧠 Analyze Graph"):
                with st.spinner("Analyzing..."):
                    prompt = build_ai_prompt(df_3d, axis_x, axis_y, axis_z, current_filters)
                    st.write(query_ollama(prompt))

            # 2. Pergunta Personalizada (Restaurado)
            st.markdown("---")
            with st.expander("💬 Ask a Custom Question"):
                st.caption("Ask specific details about this graph slice.")
                user_q = st.text_area("Your Question:")
                
                if st.button("Submit Question"):
                    if user_q:
                        with st.spinner("Thinking..."):
                            # Constrói o prompt base com os dados + a pergunta do usuário
                            base_prompt = build_ai_prompt(df_3d, axis_x, axis_y, axis_z, current_filters)
                            full_prompt = base_prompt + f"\n\nUSER FOLLOW-UP QUESTION: {user_q}\nPLEASE FOCUS ON ANSWERING THIS QUESTION BASED ON THE DATA ABOVE."
                            
                            # Exibe a resposta
                            st.write(query_ollama(full_prompt))
                    else:
                        st.warning("Please type a question first.")

    with tab_corr:
        st.subheader("Correlation Matrix (Raw Values)")
        st.caption("Correlations using NATURAL units ($, kWh, %, etc) to show real physical trade-offs.")
        
        df_num = df.select_dtypes(include=[np.number])
        corr = df_num.corr()
        
        # 1. Linhas: Apenas os Fatores de Entrada (Pesos)
        rows = [c for c in cols_factors if c in corr.index]
        
        # 2. Colunas: Prioridade para DADOS BRUTOS
        # Regra: Pega tudo que é numérico, EXCETO os Scores individuais, IDs e os próprios pesos
        cols_raw = [c for c in df.columns 
                    if 'Score_' not in c 
                    and c not in cols_factors 
                    and c not in ['Sim_ID', 'Timestamp', 'GLOBAL_PERFORMANCE_INDEX']
                    and df[c].dtype in ['float64', 'int64']]
        
        # Mantemos o GLOBAL_PERFORMANCE_INDEX pois ele é o resultado final importante
        # Build pairs [Raw, Score, Raw, Score, ...]
        target_cols = []
        for raw_metric in cols_raw:
            target_cols.append(raw_metric)           # Raw
            score_name = f'Score_{raw_metric}'
            if score_name in df.columns:
                target_cols.append(score_name)       # Score pair
        target_cols.append('GLOBAL_PERFORMANCE_INDEX')  # Global no final

        
        # Garante que as colunas existem na matriz de correlação
        valid_cols = [c for c in target_cols if c in corr.columns]
        
        if rows and valid_cols:
            fig_hm = px.imshow(
                corr.loc[rows, valid_cols], 
                text_auto=".2f", 
                aspect="auto", 
                color_continuous_scale="RdBu_r", # Vermelho/Azul Divergente
                zmin=-1, 
                zmax=1
            )
            st.plotly_chart(fig_hm, width="stretch")
            
            st.info("""
            **Interpretation Guide (Natural Correlation):**
            * 🟥 **Positive (Red):** As Weight increases, the Metric value **INCREASES**.
            * 🟦 **Negative (Blue):** As Weight increases, the Metric value **DECREASES**.
            
            *Example:* If 'W_Grid' has a **Negative** correlation with 'Tech_Losses', it means investing in the Grid **reduces** losses (Physical Reality).
            """)
        else:
            st.warning("Insufficient data for correlation.")
        
        st.markdown("---")
        st.subheader("📚 Indicator Reference Guide")
        st.markdown("Full list of definitions and mathematical formulas for all metrics used above.")

        # Itera sobre o dicionário METRIC_INFO para listar tudo em sequência
        for metric, info in METRIC_INFO.items():
            # Cria um container visualmente separado para cada indicador
            with st.container():
                c_text, c_math = st.columns([1.5, 1])
                
                with c_text:
                    st.markdown(f"### 📌 {metric}")
                    st.markdown(f"**🇬🇧 En:** {info['en']}")
                    st.markdown(f"**🇫🇷 Fr:** {info['fr']}")
                
                with c_math:
                    # Exibe a fórmula centralizada verticalmente
                    st.latex(info['eq'])
                
                # Linha divisória sutil entre os itens
                st.divider()

    with tab_zones:
        st.header("Zonal Analysis: Distribution & Profitability")
        st.caption("Understand how incentives alter the network geography (Rural vs Urban).")
        
        if 'Cap_Pct_Rural' in df.columns:

            col_dist, col_heat = st.tabs(["📊 Capacity Distribution", "💰 Revenue Heatmap ($/kW)"])

            # --- GRAPH 1: DISTRIBUTION (Stacked Area) ---
            with col_dist:
                st.subheader(f"Allocation Evolution (Rural/Mixed/Urban) vs {axis_x}")
                
                # Check for necessary columns
                cols_zone = ['Cap_Pct_Rural', 'Cap_Pct_Mixed', 'Cap_Pct_Urban']
                
                # FALLBACK IF COLUMNS MISSING (for visualization demo)
                if 'Cap_Pct_Rural' not in df_3d.columns and 'Social_Equity_Pct' in df_3d.columns:
                    df_3d['Cap_Pct_Rural'] = df_3d['Social_Equity_Pct']
                    df_3d['Cap_Pct_Rest'] = 100 - df_3d['Cap_Pct_Rural']
                    df_3d['Cap_Pct_Mixed'] = df_3d['Cap_Pct_Rest'] * 0.6 
                    df_3d['Cap_Pct_Urban'] = df_3d['Cap_Pct_Rest'] * 0.4 

                # Filter and Sort for the Line Chart
                df_line = df_3d.sort_values(by=axis_x)

                if all(c in df_line.columns for c in cols_zone):
                    fig_area = px.area(
                        df_line, 
                        x=axis_x, 
                        y=cols_zone,
                        labels={'value': 'Installed Capacity (%)', 'variable': 'Zone', axis_x: axis_x},
                        color_discrete_map={
                            'Cap_Pct_Rural': '#2ecc71', # Green
                            'Cap_Pct_Mixed': '#f1c40f', # Yellow
                            'Cap_Pct_Urban': '#e74c3c'  # Red
                        }
                    )
                    fig_area.update_layout(yaxis_range=[0, 100], hovermode="x unified")
                    st.plotly_chart(fig_area, width="stretch")
                    
                    st.info(f"💡 **How to read:** If the **Green** area grows as you move to the right, the **{axis_x}** incentive is successfully attracting Rural investment.")
                else:
                    st.warning("⚠️ Columns 'Cap_Pct_...' not found in CSV. Re-run the Batch script with zonal calculations enabled.")

            # --- GRAPH 2: REVENUE HEATMAPS (Side by Side) ---
            with col_heat:
                st.subheader("Profitability Comparison ($/kW)")
                st.caption("Red = Low Return | Green = High Return (Gold Mine)")

                # Secondary Y-Axis Selector for Heatmap
                hm_y = st.selectbox("Secondary Y-Axis for Heatmap", filter_vars, index=0)
                    
                # Filter for Heatmap (Remove the HM Y filter)
                mask_hm = pd.Series([True] * len(df))
                for f in [c for c in filter_vars if c != hm_y]:
                    mask_hm &= (df[f] == current_filters[f])
                df_hm = df[mask_hm]

                cols_rev = ['Rev_per_kW_Rural', 'Rev_per_kW_Mixed', 'Rev_per_kW_Urban']
                max_val = df_hm[cols_rev].max().max()
                
                c1, c2, c3 = st.columns(3)
                for i, (col, title) in enumerate(zip(cols_rev, ["Rural", "Mixed", "Urban"])):
                    with [c1, c2, c3][i]:
                        st.markdown(f"**{title}**")
                        pivot = df_hm.pivot_table(index=hm_y, columns=axis_x, values=col)
                        fig = px.imshow(pivot, color_continuous_scale="RdYlGn", origin='lower', zmin=0, zmax=max_val, labels=dict(color="$/kW"))
                        fig.update_layout(height=300, margin=dict(l=0, r=0, t=0, b=0))
                        st.plotly_chart(fig, width="stretch")
                       
        else:
            st.error("Missing Zonal Data.")
            st.info("Ensure 'MASTER_Bus_Results.csv' and 'Rede_240Bus_Dados.xlsx' are in the correct paths.")

# ==============================================================================
# EXPORT ALL GRAPHS & TABLES TO XLS (NEW SECTION)
# ==============================================================================
def export_all_to_xls(df, df_3d, axis_x, axis_y, axis_z, current_filters, cols_factors):
    """Export all graphs/tables data to single XLS with one sheet per visualization."""
    with st.spinner("📊 Exporting all graphs to Excel..."):
        output_file = os.path.join(RESULTS_DIR, f"Visualizer_Export_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.xlsx")
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # 1. RAW DATA (Base)
            df.to_excel(writer, sheet_name='01_Raw_Scores', index=False)
            
            # 2. 3D LANDSCAPE DATA
            df_3d_export = df_3d[[axis_x, axis_y, axis_z]].copy()
            df_3d_export.to_excel(writer, sheet_name='02_3D_Data', index=False)
            
            # 3. CORRELATION MATRIX (Full)
            df_num = df.select_dtypes(include=[np.number])
            corr_matrix = df_num.corr()
            corr_df = corr_matrix.round(3)
            corr_df.to_excel(writer, sheet_name='03_Correlation_Full')
            
            # 4. TOP CORRELATIONS (Input Weights only)
            rows = [c for c in cols_factors if c in corr_matrix.index]
            cols_raw = [c for c in df.columns if 'Score_' not in c and c not in cols_factors and c not in ['Sim_ID']]
            valid_cols = [c for c in cols_raw if c in corr_matrix.columns]
            top_corr = corr_matrix.loc[rows, valid_cols].round(3)
            top_corr.to_excel(writer, sheet_name='04_Top_Correlations')
            
            # 5. ZONAL DATA
            zone_cols = [c for c in df.columns if any(z in c for z in ['Rural', 'Mixed', 'Urban', 'Cap_Pct', 'Rev_per_kW'])]
            if zone_cols:
                df[df.columns.intersection(zone_cols)].to_excel(writer, sheet_name='05_Zonal_Data', index=False)
            
            # 6. METRIC DESCRIPTIONS (Info Table)
            metrics_df = pd.DataFrame([
                {'Metric': k, 'English': v['en'], 'Formula': v['eq']} 
                for k, v in METRIC_INFO.items()
            ])
            metrics_df.to_excel(writer, sheet_name='06_Metric_Info', index=False)
            
            # 7. FILTER SUMMARY
            filters_df = pd.DataFrame([current_filters])
            filters_df.to_excel(writer, sheet_name='07_Filters_Used', index=False)
        
        st.sidebar.success(f"✅ **Exported!** [{output_file}]")
        st.sidebar.markdown(f"[📁 **Download Here**](data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{base64.b64encode(open(output_file, 'rb').read()).decode()})")
        return output_file




if __name__ == "__main__":
    main()


# streamlit run "./code/visualizer.py"    