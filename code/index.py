import numpy as np
import networkx as nx
import os
import time
import sys
import os
import pandas as pd
from config import CFG

# ==============================================================================
# CONFIGURATION (USANDO O CONFIG.PY QUE ARRUMAMOS)
# ==============================================================================
# Usa as variáveis inteligentes do config.py
DIR_RESULTADOS = CFG["OUTPUT_DIR"]
MODEL_DIR = os.path.dirname(CFG["DSS_FILE"])

# Caminho do Excel da Rede
ARQUIVO_EXCEL_REDE = os.path.join(MODEL_DIR, "Rede_240Bus_Dados.xlsx")

# Horas consideradas como Pico
PEAK_HOURS = [15, 16, 17, 18]

# Se receber argumentos via linha de comando (do dashboard), usa eles
if len(sys.argv) > 1:
    try:
        arg_str = sys.argv[1]
        PEAK_HOURS = [int(x) for x in arg_str.split(',')]
        print(f"[INFO] Peak hours set from argument: {PEAK_HOURS}")
    except Exception as e:
        print(f"[INFO] Failed to parse peak hours. Using default: {PEAK_HOURS}")
else:
    print(f"[INFO] Using default peak hours: {PEAK_HOURS}")

# Definição dos Arquivos (Baseado no diretório seguro)
FILE_BUS_RES  = os.path.join(DIR_RESULTADOS, "MASTER_Bus_Results.csv")
FILE_HOURLY   = os.path.join(DIR_RESULTADOS, "MASTER_Hourly_Results.csv")
FILE_VOLTAGE  = os.path.join(DIR_RESULTADOS, "MASTER_Voltage_Log.csv")
FILE_OUTPUT   = os.path.join(DIR_RESULTADOS, "RELATORIO_COMPLETO_SCORES.csv")

# Verifica se os arquivos existem antes de começar
if not os.path.exists(FILE_HOURLY):
    print(f"❌ CRITICAL ERROR: Input file not found: {FILE_HOURLY}")
    print(f"   I was looking at: {DIR_RESULTADOS}")
    sys.exit(1)

# ==============================================================================
# 2. FUNÇÕES AUXILIARES DE LIMPEZA E CÁLCULO
# ==============================================================================

def clean_name(name):
    """Padroniza nomes removendo prefixos/sufixos para garantir conexões."""
    n = str(name).lower().strip()
    for tag in ['t_', 'load_', '_l', '_node', 'source', 'eq_']: 
        n = n.replace(tag, '')
    return n

def calcular_gini(array):
    """Calcula o Coeficiente de Gini (Desigualdade)."""
    array = np.array(array, dtype=np.float64)
    if np.amin(array) < 0: return 0 
    array += 0.0000001 # Evita div por zero
    array = np.sort(array)
    index = np.arange(1, array.shape[0] + 1)
    n = array.shape[0]
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array)))

def calcular_scores_normalizados(df):
    """
    Cria colunas de Score (0-100) para cada indicador.
    Inverte a lógica para custos/riscos (onde menor é melhor).
    """
    df_norm = df.copy()
    
    # Define a direção: True = Maior é Melhor | False = Menor é Melhor
    direcao_indicadores = {
        # Financeiro
        'ROI_10y_Pct': True, 
        'LCOE_USD_kWh': False, 
        'Windfall_Profit_Index': False, # Lucro excessivo = Ruim para política pública
        'Revenue_Volatility': False,
        
        # Técnico
        'Grid_Congestion_Max_Pct': False, 
        'Tech_Losses_Total_kWh': False, 
        'Voltage_Deviation_Avg': False,
        'Self_Sufficiency_Pct': True, 
        'Peak_Coincidence_Pct': True,
        
        # Sócio-Ambiental & Política
        'Social_Equity_Pct': True, 
        'CO2_Avoided_Tons_Year': True,
        'Subsidy_Intensity': False, # Menor custo fiscal = Melhor
        'Incentivized_Carbon_Cost': False, 
        'Private_Investment_Leverage': True, 
        'Revenue_Disparity_Ratio': True, 
        'Gini_Incentivo': False # Gini alto = Desigualdade = Ruim
    }

    cols_score = []

    print("Calculating Normalized Scores (0-100)...")
    
    for col, maior_eh_melhor in direcao_indicadores.items():
        if col in df.columns:
            min_val = df[col].min()
            max_val = df[col].max()
            delta = max_val - min_val
            
            new_col_name = f"Score_{col}"
            
            if delta == 0:
                df_norm[new_col_name] = 50.0 # Neutro se não houve variação
            else:
                if maior_eh_melhor:
                    # (Valor - Min) / Delta -> 100 é o Máximo
                    df_norm[new_col_name] = ((df[col] - min_val) / delta) * 100
                else:
                    # (Max - Valor) / Delta -> 100 é o Mínimo (Invertido)
                    df_norm[new_col_name] = ((max_val - df[col]) / delta) * 100
            
            cols_score.append(new_col_name)

    # Score Global (Média de todos os scores)
    df_norm['GLOBAL_PERFORMANCE_INDEX'] = df_norm[cols_score].mean(axis=1)
    
    return df_norm

# ==============================================================================
# 3. CARREGAMENTO DE DADOS
# ==============================================================================

def carregar_dados():
    print("Loading data files...")
    try:
        df_hourly = pd.read_csv(FILE_HOURLY)
        df_bus = pd.read_csv(FILE_BUS_RES)
        
        if os.path.exists(FILE_VOLTAGE):
            df_volt = pd.read_csv(FILE_VOLTAGE)
        else:
            df_volt = pd.DataFrame()
            print("Aviso: Arquivo de Tensão não encontrado (usando 0).")
        
        # Lê e Limpa Excel da Rede
        df_lines = pd.read_excel(ARQUIVO_EXCEL_REDE, sheet_name="Lines")
        df_lines['Bus1'] = df_lines['Bus1'].apply(clean_name)
        df_lines['Bus2'] = df_lines['Bus2'].apply(clean_name)
        
        df_transf = pd.read_excel(ARQUIVO_EXCEL_REDE, sheet_name="Transformers")
        # Ajuste inteligente de colunas do trafo
        c1 = 'Bus1_High' if 'Bus1_High' in df_transf.columns else 'Bus1'
        c2 = 'Bus2_Low' if 'Bus2_Low' in df_transf.columns else 'Bus2'
        df_transf['Bus1'] = df_transf[c1].apply(clean_name)
        df_transf['Bus2'] = df_transf[c2].apply(clean_name)

        df_buses_ref = pd.read_excel(ARQUIVO_EXCEL_REDE, sheet_name="Buses")
        df_buses_ref['BusName'] = df_buses_ref['BusName'].apply(clean_name)
        
        df_loads_ref = pd.read_excel(ARQUIVO_EXCEL_REDE, sheet_name="Loads")
        df_loads_ref['Bus'] = df_loads_ref['Bus'].apply(clean_name)

        # Limpa Resultados também
        df_bus['Bus'] = df_bus['Bus'].apply(clean_name)

        # Limpeza de espaços em nomes de colunas
        df_hourly.columns = df_hourly.columns.str.strip()
        
        print(f"Data loaded! {df_hourly['Sim_ID'].nunique()} scenarios identified.")
        return df_hourly, df_bus, df_volt, df_lines, df_transf, df_buses_ref, df_loads_ref
    except Exception as e:
        print(f"Critical error loading files: {e}")
        return None

def montar_grafo_rede(df_lines, df_transf):
    """Monta a topologia elétrica conectada para cálculo de fluxo."""
    G = nx.DiGraph()
    
    # Linhas
    for _, row in df_lines.iterrows():
        r = row.get('R_Total_Ohms_Calc', 0.05)
        lim = row.get('NormAmps', 400)
        G.add_edge(row['Bus1'], row['Bus2'], r=r, limit=lim)
        
    # Transformadores
    for _, row in df_transf.iterrows():
        G.add_edge(row['Bus1'], row['Bus2'], r=0.01, limit=2000)
        
    # Links Virtuais (Correção de Topologia Específica)
    links = [('bus1', 'bus1001'), ('bus1', 'bus2001'), ('bus1', 'bus3001')]
    for u, v in links:
        G.add_edge(clean_name(u), clean_name(v), r=0.001, limit=9999)

    # Identifica Fonte (Raiz)
    possiveis = ['bus_xfmr', 'substation', 'bus1', 'sourcebus', 'eq_source_bus'] 
    possiveis = [clean_name(x) for x in possiveis]
    
    fonte = next((n for n in possiveis if n in G.nodes()), None)
    
    # Fallback: Nó com maior grau de saída
    if not fonte:
        try: fonte = max(dict(G.out_degree()).items(), key=lambda x: x[1])[0]
        except: pass

    if fonte:
        try:
            tree = nx.bfs_tree(G, source=fonte)
            nos_reverso = list(reversed(list(nx.topological_sort(tree))))
            print(f"Network graph constructed. Source node: {fonte}")
            return G, tree, nos_reverso
        except: 
            print("Error: Network disconnected or cyclic.")
            return G, None, None
            
    print("Error: Source node not found.")
    return G, None, None

# ==============================================================================
# 4. MOTOR PRINCIPAL DE CÁLCULO
# ==============================================================================

def main():
    dados = carregar_dados()
    if not dados: return
    df_hourly, df_bus_res, df_volt, df_lines, df_transf, df_buses_ref, df_loads_ref = dados
    
    # Prepara Topologia
    G, tree, nos_reverso = montar_grafo_rede(df_lines, df_transf)
    
    # Mapeamentos
    mapa_zonas = dict(zip(df_buses_ref['BusName'], df_buses_ref['Zone']))
    mapa_carga_base = df_loads_ref.groupby('Bus')['kW'].sum().to_dict()
    carga_total_sistema = sum(mapa_carga_base.values())
    
    resultados = []
    simulacoes = df_hourly['Sim_ID'].unique()
    total = len(simulacoes)
    
    print(f"Processando {total} cenários...")
    start_t = time.time()
    
    for idx, sim_id in enumerate(simulacoes):
        if idx % 50 == 0: print(f"   Processed {idx}/{total} scenarios...")
        
        # Filtra dados do cenário
        ts = df_hourly[df_hourly['Sim_ID'] == sim_id]
        bus = df_bus_res[df_bus_res['Sim_ID'] == sim_id]
        
        if ts.empty: continue
        meta = ts.iloc[0] # Metadados estão repetidos nas linhas
        
        # --- PARÂMETROS ---
        w_grid = meta.get('Weight_Grid', 0)
        w_cap = meta.get('Weight_Cap', 0)
        w_soc = meta.get('Weight_Social', 0)
        w_env = meta.get('Weight_Env', 0)
        
        capex = meta.get('Total_Capex', 0)
        receita_dia = meta.get('Total_Revenue', 0)
        wacc = meta.get('WACC', 0.05)
        
        # Geração e Carga
        gen_mwh_yr = (ts['PV_Total'].sum() + ts['Wind_Total'].sum()) * 365 / 1000.0
        load_dia = ts['Carga_Total'].sum()
        
        # Dicionário de KPIs
        kpis = {
            'Sim_ID': sim_id,
            'W_Grid': w_grid, 'W_Cap': w_cap, 'W_Social': w_soc, 'W_Env': w_env,
            'Total_Capex': capex, 'Total_Revenue_Daily': receita_dia
        }

        # ---------------------------------------------------------
        # 1. INDICADORES FINANCEIROS
        # ---------------------------------------------------------
        receita_10y = receita_dia * 365 * 10
        if capex > 0:
            kpis['ROI_10y_Pct'] = (receita_10y - capex) / capex * 100
            kpis['LCOE_USD_kWh'] = ((capex/20) + capex*0.01) / (gen_mwh_yr * 1000) if gen_mwh_yr > 0 else 0
            
            # Windfall Profit (Sobrelucro acima do WACC)
            kpis['Windfall_Profit_Index'] = (kpis['ROI_10y_Pct'] - (wacc*100)) / (wacc*100) if wacc > 0 else 0
        else:
            kpis['ROI_10y_Pct'] = 0
            kpis['LCOE_USD_kWh'] = 0
            kpis['Windfall_Profit_Index'] = 0

        # Volatilidade da Receita
        # Tenta pegar as colunas de receita individuais se existirem, senão estima
        cols_rev = [c for c in ts.columns if 'Rev_' in c]
        if cols_rev:
            rev_horaria = ts[cols_rev].sum(axis=1)
            media_h = rev_horaria.mean()
            kpis['Revenue_Volatility'] = (rev_horaria.std() / media_h * 100) if media_h > 0 else 0
        else:
            kpis['Revenue_Volatility'] = 0

        # ---------------------------------------------------------
        # 2. INDICADORES TÉCNICOS BÁSICOS
        # ---------------------------------------------------------
        kpis['Self_Sufficiency_Pct'] = (gen_mwh_yr * 1000 / (load_dia * 365) * 100) if load_dia > 0 else 0
        
        # Coincidência de Pico
        ts_pico = ts[ts['Hora'].isin(PEAK_HOURS)]
        if not ts_pico.empty:
            inj_pico = ts_pico['DER_Injecao_Liq'].sum()
            load_pico = ts_pico['Carga_Total'].sum()
            kpis['Peak_Coincidence_Pct'] = (inj_pico / load_pico * 100) if load_pico > 0 else 0
        else:
            kpis['Peak_Coincidence_Pct'] = 0

        # Desvio de Tensão
        if not df_volt.empty:
            volt_data = df_volt[df_volt['Sim_ID'] == sim_id]
            if not volt_data.empty:
                # Pega colunas numéricas de tensão (ignora metadados)
                cols_v = [c for c in volt_data.columns if c not in meta.index and 'bus' in c.lower()]
                vals = volt_data[cols_v].values
                kpis['Voltage_Deviation_Avg'] = np.nanmean(np.abs(vals - 1.0))
            else: kpis['Voltage_Deviation_Avg'] = 0
        else: kpis['Voltage_Deviation_Avg'] = 0

        # ---------------------------------------------------------
        # 3. INDICADORES SÓCIO-AMBIENTAIS & POLÍTICA
        # ---------------------------------------------------------
        kpis['CO2_Avoided_Tons_Year'] = gen_mwh_yr * 0.286670 # 632lb CO2/MWh = 0.286670 ton CO2/MWh
        
        # Equidade Social e Disparidade
        cap_total = bus['Capacity (kW)'].sum()
        if cap_total > 0:
            bus_zone = bus.copy()
            bus_zone['Zone'] = bus_zone['Bus'].map(mapa_zonas).fillna('Mixed')
            cap_rural = bus_zone[bus_zone['Zone'] == 'Rural']['Capacity (kW)'].sum()
            cap_urban = bus_zone[bus_zone['Zone'] == 'Urban']['Capacity (kW)'].sum()
            
            kpis['Social_Equity_Pct'] = (cap_rural / cap_total * 100)
            
            # Gini (Concentração de Capacidade)
            vals = bus['Capacity (kW)'].values
            kpis['Gini_Incentivo'] = calcular_gini(vals[vals > 0])
            
            # Disparidade de Receita (Estimativa)
            rev_social_total = ts['Rev_Social'].sum() if 'Rev_Social' in ts.columns else 0
            rev_base_total = receita_dia - rev_social_total
            
            rev_per_kw_base = rev_base_total / cap_total
            rev_per_kw_rural = rev_per_kw_base + (rev_social_total / cap_rural) if cap_rural > 0 else rev_per_kw_base
            
            kpis['Revenue_Disparity_Ratio'] = rev_per_kw_rural / rev_per_kw_base if rev_per_kw_base > 0 else 1.0
        else:
            kpis['Social_Equity_Pct'] = 0
            kpis['Gini_Incentivo'] = 0
            kpis['Revenue_Disparity_Ratio'] = 1.0

        # Eficiência do Subsídio
        rev_energia = ts['Rev_Energia'].sum() if 'Rev_Energia' in ts.columns else 0
        subsidio_total = receita_dia - rev_energia
        
        kpis['Subsidy_Intensity'] = (subsidio_total * 365) / gen_mwh_yr if gen_mwh_yr > 0 else 0
        
        incentivo_10y = subsidio_total * 365 * 10
        kpis['Private_Investment_Leverage'] = capex / incentivo_10y if incentivo_10y > 0 else 0
        
        # Custo Carbono Incentivado
        # Estima parte ambiental do subsídio pelo peso W_ENV
        w_sum = w_soc + w_env + w_grid + w_cap
        share_env = w_env / w_sum if w_sum > 0 else 0
        subsidio_amb = subsidio_total * share_env
        kpis['Incentivized_Carbon_Cost'] = (subsidio_amb * 365) / kpis['CO2_Avoided_Tons_Year'] if kpis['CO2_Avoided_Tons_Year'] > 0 else 0

        # ---------------------------------------------------------
        # 4. TÉCNICO AVANÇADO (FLUXO - BACKWARD SWEEP)
        # ---------------------------------------------------------
        if nos_reverso and not ts.empty and carga_total_sistema > 0:
            idx_max = ts['Carga_Total'].idxmax()
            row_pico = ts.loc[idx_max]
            
            fat_load = row_pico['Carga_Total'] / carga_total_sistema
            fat_gen = (row_pico['PV_Total'] + row_pico['Wind_Total']) / cap_total if cap_total > 0 else 0
            
            S_inj = {n: 0.0 for n in G.nodes()}
            cap_map = bus.groupby('Bus')['Capacity (kW)'].sum().to_dict()
            
            for no in G.nodes():
                l = mapa_carga_base.get(no, 0.0) * fat_load
                g = cap_map.get(no, 0.0) * fat_gen
                S_inj[no] = l - g 
            
            max_load = 0
            loss_pico = 0
            V_base = 7967 * 1.732 
            
            for no in nos_reverso:
                preds = list(tree.predecessors(no))
                if not preds: continue
                pai = preds[0]
                S_inj[pai] += S_inj[no]
                
                fluxo = abs(S_inj[no]) * 1000
                
                if G.has_edge(pai, no):
                    dados_aresta = G[pai][no]
                    r = dados_aresta.get('r', 0.01)
                    limite = dados_aresta.get('limit', 9999)
                else:
                    r, limite = 0.01, 9999

                I = fluxo / V_base
                loss_pico += 3 * r * (I**2) / 1000 
                if limite > 0:
                    load = (I / limite) * 100
                    if load > max_load: max_load = load
            
            kpis['Grid_Congestion_Max_Pct'] = max_load
            kpis['Tech_Losses_Total_kWh'] = loss_pico * 24 * 0.5 
        else:
            kpis['Grid_Congestion_Max_Pct'] = 0
            kpis['Tech_Losses_Total_kWh'] = 0
            
        resultados.append(kpis)

    # ==============================================================================
    # 5. NORMALIZAÇÃO E EXPORTAÇÃO
    # ==============================================================================
    df_final = pd.DataFrame(resultados)
    
    # Aplica Normalização (0-100) e Score Global
    df_final = calcular_scores_normalizados(df_final)
    
    # Reordena para ficar organizado
    cols_meta = ['Sim_ID', 'W_Grid', 'W_Cap', 'W_Social', 'W_Env', 'GLOBAL_PERFORMANCE_INDEX']
    cols_metrics = [c for c in df_final.columns if c not in cols_meta and 'Score_' not in c]
    cols_scores = [c for c in df_final.columns if 'Score_' in c]
    
    final_order = cols_meta + cols_metrics + cols_scores
    # Garante que só usa colunas que existem
    final_order = [c for c in final_order if c in df_final.columns]
    
    df_final = df_final[final_order]
    
    # Salva
    df_final.to_csv(FILE_OUTPUT, index=False, float_format="%.4f")
    
    print("\n" + "="*60)
    print(f"COMPLETE REPORT GENERATED!")
    print(f"File: {FILE_OUTPUT}")
    print(f"Total Scenarios: {len(df_final)}")
    print(f"Columns: {len(df_final.columns)}")
    print("="*60)

if __name__ == "__main__":
    main()

    # streamlit run "c:/Users/KKCOD/OneDrive - Université Laval/Recherche/codes/code/index.py"