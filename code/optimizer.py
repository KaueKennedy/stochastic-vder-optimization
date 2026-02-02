import pandas as pd
import numpy as np
import networkx as nx
from docplex.mp.model import Model
from config import CFG 
import os

# ==============================================================================
# CONFIGURATION (CAMINHOS DINÂMICOS E BLINDADOS)
# ==============================================================================
# 1. Descobre onde o arquivo 'visualizer.py' está salvo no disco
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Sobe um nível para chegar na pasta raiz do projeto (pasta 'codes')
# Ele faz: "C:\...\code" -> "C:\...\codes"
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

# 3. Agora montamos os caminhos a partir da raiz segura
# (Não importa onde você dê o Play, ele sempre vai acertar)
DATA_FILE = os.path.join(PROJECT_ROOT, "resultados", "RELATORIO_COMPLETO_SCORES.csv")
BUS_FILE = os.path.join(PROJECT_ROOT, "resultados", "MASTER_Bus_Results.csv")

# Excel da Rede (Note como é fácil navegar nas pastas agora)
EXCEL_REDE = os.path.join(PROJECT_ROOT, "Iowa_Distribution_Test_Systems", "OpenDSS Model", "OpenDSS Model", "Rede_240Bus_Dados.xlsx")

# Arquivos de Contexto
CONTEXT_FILES = {
    "Hourly": os.path.join(PROJECT_ROOT, "resultados", "MASTER_Hourly_Results.csv"),
    "Bus": BUS_FILE,
    "Voltage": os.path.join(PROJECT_ROOT, "resultados", "MASTER_Voltage_Log.csv"),
    "History": os.path.join(PROJECT_ROOT, "resultados", "Optimization_History_Log.csv")
}

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def normalize_name(name):
    if pd.isna(name): return ""
    return str(name).strip().lower()

def calculate_daily_pmt(capex_total, years, rate):
    """Calculates daily amortized cost (PMT/365)."""
    if rate <= 0.001: return (capex_total / max(1, years)) / 365.0
    factor = (rate * (1 + rate)**years) / ((1 + rate)**years - 1)
    return (capex_total * factor) / 365.0

# ==============================================================================
# 1. TOPOLOGY BUILDER
# ==============================================================================
# [EM optimizer.py] Substitua a função build_network por esta:

def build_network(lines_df, transformers_df, buses_df, s_base_mva=10.0):
    print("   [TOPOLOGY] Building network model with Coordinates...")
    bus_kv_map = {}
    bus_zone_map = {}
    bus_coords = {} # <--- NOVA VARIÁVEL PARA GUARDAR X,Y
    
    if buses_df is not None:
        for _, row in buses_df.iterrows():
            nm = normalize_name(row.get('BusName') or row.get('Bus'))
            if nm:
                try: bus_kv_map[nm] = float(row.get('kVBase') or row.get('kV_base') or 12.47)
                except: bus_kv_map[nm] = 12.47
                bus_zone_map[nm] = str(row.get('Zone', 'Mixed')).strip()
                
                # --- CAPTURA DE COORDENADAS (Tenta X/Y ou Lat/Lon) ---
                try:
                    x = float(row.get('x') or row.get('X') or row.get('Longitude') or 0.0)
                    y = float(row.get('y') or row.get('Y') or row.get('Latitude') or 0.0)
                    if x != 0 or y != 0:
                        bus_coords[nm] = (x, y)
                except: pass
                # ------------------------------------------------------
    
    topo_nodes = set(bus_kv_map.keys())
    branches = []

    # (O resto do código de linhas e transformadores continua igual...)
    if lines_df is not None:
        for _, row in lines_df.iterrows():
            try:
                u, v = normalize_name(row.get('Bus1')), normalize_name(row.get('Bus2'))
                if not u or not v: continue
                topo_nodes.add(u); topo_nodes.add(v)
                r = float(row.get('R_Total_Ohms_Calc') or 0.1)
                x = float(row.get('X_Total_Ohms_Calc') or 0.1)
                raw_amps = row.get('NormAmps')
                if pd.isna(raw_amps) or float(raw_amps) <= 0: i_max = 9999.0
                else: i_max = float(raw_amps)
                kv = bus_kv_map.get(u, 12.47)
                z_base = (kv**2)/s_base_mva
                branches.append({
                    'id': f"L_{u}_{v}", 'from': u, 'to': v,
                    'x_pu': max(1e-5, x/z_base),
                    'p_max': (np.sqrt(3)*kv*i_max/1000)/s_base_mva,
                    'type': 'line'
                })
            except: pass

    if transformers_df is not None:
        name_counter = {}
        for _, row in transformers_df.iterrows():
            try:
                u, v = normalize_name(row.get('Bus1_High')), normalize_name(row.get('Bus2_Low'))
                if not u or not v: continue
                topo_nodes.add(u); topo_nodes.add(v)
                kva = float(row.get('kVA') or 5000)
                xhl = float(row.get('XHL_Percent') or 1.0)
                s_rated = kva/1000.0
                base_name = str(row.get('Name'))
                if not base_name or base_name == 'nan': base_name = f"T_{u}_{v}"
                clean_name = "".join(c for c in base_name if c.isalnum() or c in "_-")
                if clean_name in name_counter: name_counter[clean_name] += 1; unique_id = f"{clean_name}_{name_counter[clean_name]}"
                else: name_counter[clean_name] = 0; unique_id = clean_name
                branches.append({'id': unique_id, 'from': u, 'to': v,
                    'x_pu': max(1e-5, (xhl/100)*(s_base_mva/s_rated)),
                    'p_max': s_rated/s_base_mva, 'type': 'xfmr'})
            except: pass

    links = [('bus1', 'bus1001'), ('bus1', 'bus2001'), ('bus1', 'bus3001')]
    existing = set((b['from'], b['to']) for b in branches)
    for u, v in links:
        if (u,v) not in existing:
            branches.append({'id':f"SW_{u}_{v}", 'from':u, 'to':v, 'x_pu':1e-4, 'p_max':9999.0, 'type':'sw'})
            topo_nodes.add(u); topo_nodes.add(v)

    # ==========================================================================
    # 🔌 AUTO-CONNECT: CORREÇÃO DE ILHAMENTO (CRUCIAL!)
    # ==========================================================================
    # Isso substitui a leitura da aba 'Transformers' que estava falhando.
    # Ele recria a conexão entre a Média Tensão (bus1003) e a Carga (t_bus1003_l)
    
    print("   [TOPOLOGY] Auto-connecting High Voltage (bus) -> Low Voltage (t_bus)...")
    count_auto = 0
    existing_links = set((b['from'], b['to']) for b in branches)
    
    # Cria lista estática para não afetar o loop
    node_list = list(topo_nodes) 
    
    for n in node_list:
        # Detecta nós de carga pelo padrão de nome (ex: t_bus1003_l)
        if n.startswith("t_") and n.endswith("_l"):
            # Deduz o nome do pai: remove "t_" (início) e "_l" (fim)
            # t_bus1003_l  -->  bus1003
            parent_node = n[2:-2] 
            
            # Se o pai existe na rede principal
            if parent_node in topo_nodes:
                # Se ainda não existe conexão, CRIA AGORA
                if (parent_node, n) not in existing_links and (n, parent_node) not in existing_links:
                    branches.append({
                        'id': f"XF_AUTO_{parent_node}", 
                        'from': parent_node, 
                        'to': n,
                        'x_pu': 0.001,      # Impedância mínima (quase zero)
                        'p_max': 99999.0,   # Fluxo infinito permitido
                        'type': 'xfmr'
                    })
                    count_auto += 1
                    existing_links.add((parent_node, n))

    print(f"   ✅ {count_auto} Transformadores recriados automaticamente.")

    # RETORNA A NOVA VARIÁVEL 'bus_coords' NO FINAL
    return list(topo_nodes), branches, bus_zone_map, bus_coords

# ==============================================================================
# 2. OPTIMIZATION ENGINE
# ==============================================================================
def run_optimization(lines, buses, loads, prof, transformers=None, scenarios=None):
    print(f"--- [2] Running Stochastic Optimization (FULL PHYSICS + SCALED PRICE) ---")
    
    # 1. Parâmetros Base
    T = CFG.get("T", 24)
    S_BASE = CFG.get("S_BASE", 10.0)
    MAX_PEN = CFG.get("MAX_PENETRATION", 1.0)
    S_range = range(len(scenarios))
    ts = range(T)
    
    # --- FINANCE ---
    costs = CFG.get("COSTS", {})
    remun = CFG.get("REMUNERATION", {})
    wacc = costs.get("ECON", {}).get("WACC", 0.08)
    
    # Cálculo de Capex Diário (Amortização)
    c_pv = calculate_daily_pmt(costs["PV"]["CAPEX"], costs["PV"]["LIFE"], wacc) + (costs["PV"]["OM"]/365.0)
    c_wd = calculate_daily_pmt(costs["WIND"]["CAPEX"], costs["WIND"]["LIFE"], wacc) + (costs["WIND"]["OM"]/365.0)
    c_bess_p = costs["BESS"]["CAPEX_P"]
    c_bess_e = costs["BESS"]["CAPEX_E"]
    c_bess = calculate_daily_pmt(c_bess_e + c_bess_p/4.0, costs["BESS"]["LIFE"], wacc) + (costs["BESS"]["OM"]/365.0)
    deg_cost = costs["BESS"].get("DEG_COST", 0.0)
    capex_node_day = c_pv + c_wd + (c_bess * 2.0)
    
    def get_val(key): 
        v = remun.get(key, 0)
        return float(v['val']) if isinstance(v, dict) else float(v)
    
    w_env, w_social, w_grid, alpha = get_val("W_ENV"), get_val("W_SOCIAL"), get_val("W_GRID"), get_val("ALPHA")
    w_cap_val = remun.get("W_CAPACITY", {}).get("val", 0)
    alloc_map_cfg = remun.get("SOCIAL_ALLOCATION", {"Rural": 1.0, "Mixed": 0.0, "Urban": 0.0})

    # --- NETWORK & LOAD ---
    nodes, branches, bus_zone_map, bus_coords = build_network(lines, transformers, buses, S_BASE)   
    slack = nodes[0]
    for n in nodes:
        if 'source' in n or 'sub' in n: 
            slack = n
            break
        
    L_P = {n: np.zeros(T) for n in nodes}
    SOMA_BRUTA_EXCEL = 0.0  
        # ==========================================================================
    # 🕵️ DEBUG DE CONECTIVIDADE (Cole isso logo após build_network)
    # ==========================================================================
    print(f"\n🔌 [DIAGNÓSTICO DE REDE]")
    print(f"   Total de Nós (Nodes): {len(nodes)}")
    print(f"   Total de Ramos (Branches): {len(branches)}")
    
    if len(branches) == 0:
        print("   ❌ ERRO CRÍTICO: Nenhuma linha foi criada! A rede está desconectada.")
        print("      Verifique se as colunas 'Bus1' e 'Bus2' existem no Excel de Linhas.")
    else:
        # Verifica se os nós de carga estão conectados a alguma linha
        load_nodes = [n for n, lp in L_P.items() if sum(lp) > 0.1]
        connected_nodes = set()
        for b in branches:
            connected_nodes.add(b['from'])
            connected_nodes.add(b['to'])
            
        isolated_loads = [n for n in load_nodes if n not in connected_nodes]
        
        if isolated_loads:
            print(f"   ⚠️ ALERTA: {len(isolated_loads)} nós de carga estão ISOLADOS (sem linhas).")
            print(f"      Exemplos: {isolated_loads[:3]}")
            print("      Isso impede a exportação de energia (Causa do problema 1.18 kW).")
        else:
            print("   ✅ Conectividade OK: Todos os nós de carga possuem linhas.")
            print(f"      Exemplo de Conexão: {branches[0]['from']} <--> {branches[0]['to']}")
    
    print("="*60 + "\n")
    
    for _, r in loads.iterrows():
        b = normalize_name(r.get('Bus'))
        valor_kw_planilha = float(r['kW']) 
        SOMA_BRUTA_EXCEL += valor_kw_planilha
        if b in nodes:
            lp = ((valor_kw_planilha/1000.0)/S_BASE) * prof['load_curve']
            L_P[b] += lp

    print(f"   ℹ️  SOMA TOTAL DA PLANILHA (kW): {SOMA_BRUTA_EXCEL:.2f} kW")
      
    # --- MODEL SETUP ---
    mdl = Model(name="Stochastic_Optimization")
    mdl.parameters.mip.tolerances.mipgap = CFG["STOCHASTIC"].get("MIP_GAP", 0.001)

    # ==========================================================================
    # VARIABLES
    # ==========================================================================
    # 1º Estágio: Investimento (Independente do cenário)
    cap = mdl.continuous_var_dict(nodes, lb=0, name="cap")

    # 2º Estágio: Operação (Dependente do cenário s)
    ppv = mdl.continuous_var_dict([(t, n, s) for t in ts for n in nodes for s in S_range], lb=0, name="ppv")
    pwd = mdl.continuous_var_dict([(t, n, s) for t in ts for n in nodes for s in S_range], lb=0, name="pwd")
    pdis = mdl.continuous_var_dict([(t, n, s) for t in ts for n in nodes for s in S_range], lb=0, name="pdis")
    pch = mdl.continuous_var_dict([(t, n, s) for t in ts for n in nodes for s in S_range], lb=0, name="pch")
    soc = mdl.continuous_var_dict([(t, n, s) for t in ts for n in nodes for s in S_range], lb=0, name="soc")
    p_buy = mdl.continuous_var_dict([(t, n, s) for t in ts for n in nodes for s in S_range], lb=0, name="buy")
    p_sell = mdl.continuous_var_dict([(t, n, s) for t in ts for n in nodes for s in S_range], lb=0, name="sell")
    theta = mdl.continuous_var_dict([(t, n, s) for t in ts for n in nodes for s in S_range], lb=-3.14, ub=3.14, name="theta")
    flow = mdl.continuous_var_dict([(t, b['id'], s) for t in ts for b in branches for s in S_range], lb=-99999.0, name="flow")
    p_shed = mdl.continuous_var_dict([(t, n, s) for t in ts for n in nodes for s in S_range], lb=0, name="shed")
    viol = mdl.continuous_var_dict([(t, b['id'], s) for t in ts for b in branches for s in S_range], lb=0, name="viol")

    # ==========================================================================
    # BLOCK 1: CAPACITY, INVESTMENT & BLACKLIST
    # ==========================================================================
    BLACKLIST = ['eq_source_bus', 'bus_xfmr']
    for n in nodes:
        if n == slack or any(b in n for b in BLACKLIST):
            mdl.add_constraint(cap[n] == 0)
        else:
            mdl.add_constraint(cap[n] <= 50.0) # Limite de 50kW por nó
    
    # --- CORREÇÃO: REGRA DE PENETRAÇÃO BASEADA EM ENERGIA (kWh) ---
    # 1. Calcula energia total da carga no horizonte (kWh)
    #    (Soma das cargas de pico * Perfil normalizado da carga)
    total_load_energy = SOMA_BRUTA_EXCEL * np.sum(prof['load_curve'])
    
    # 2. Calcula potencial de geração por kW instalado (Solar + Eólica)
    #    (Soma dos perfis unitários de geração)
    unit_gen_energy = np.sum(prof['solar']) + np.sum(prof['wind'])
    
    # Evita divisão por zero caso perfis estejam zerados
    if unit_gen_energy <= 0.001: unit_gen_energy = 1.0

    # 3. Restrição: Energia Gerada <= Energia Carga * %Penetração
    #    Isso permite instalar muito mais kW se o fator de capacidade for baixo.
    mdl.add_constraint(
        mdl.sum(cap[n] for n in nodes) * unit_gen_energy <= total_load_energy * MAX_PEN
    )


    # ==========================================================================
    # BLOCK 2: OPERATIONAL CONSTRAINTS (STOCHASTIC)
    # ==========================================================================
    br_out = {n: [] for n in nodes}; br_in = {n: [] for n in nodes}
    for b in branches:
        br_out[b['from']].append(b['id']); br_in[b['to']].append(b['id'])

    for s in S_range:
        solar_s = scenarios[s]['solar']
        wind_s = scenarios[s]['wind']
        # Razão de carga cenário/base para ajuste proporcional
        load_mult_s = scenarios[s]['load'] / np.where(prof['load_curve'] > 0, prof['load_curve'], 1.0)

        for n in nodes:
            # Elo entre estágios: a operação respeita a cap instalada (1º estágio)
            for t in ts:
                mdl.add_constraint(ppv[t,n,s] <= cap[n] * solar_s[t])
                mdl.add_constraint(pwd[t,n,s] <= cap[n] * wind_s[t])
                mdl.add_constraint(pch[t,n,s] <= 0.5 * cap[n])
                mdl.add_constraint(pdis[t,n,s] <= 0.5 * cap[n])
                
                # Dinâmica da Bateria
                e_term = pch[t,n,s]*0.95 - pdis[t,n,s]/0.95
                if t == 0:
                    mdl.add_constraint(soc[t,n,s] == (2.0 * cap[n]) * 0.5 + e_term)
                else:
                    mdl.add_constraint(soc[t,n,s] == soc[t-1,n,s] + e_term)
                mdl.add_constraint(soc[t,n,s] <= 2.0 * cap[n])
            
            mdl.add_constraint(soc[T-1,n,s] == (2.0 * cap[n]) * 0.5)

        # Física da Rede por Cenário
        
        for t in ts:
            mdl.add_constraint(theta[t, slack, s] == 0)
            for b_data in branches:
                bid, u, v, x, pmax = b_data['id'], b_data['from'], b_data['to'], b_data['x_pu'], b_data['p_max']
                mdl.add_constraint(flow[t,bid,s] == (theta[t,u,s] - theta[t,v,s]) / x)
                mdl.add_constraint(flow[t,bid,s] <= pmax + viol[t,bid,s])
                mdl.add_constraint(flow[t,bid,s] >= -pmax - viol[t,bid,s])

            for n in nodes:
                der = (ppv[t,n,s] + pwd[t,n,s] + pdis[t,n,s] - pch[t,n,s]) / (S_BASE * 1000)
                grid = (p_buy[t,n,s] - p_sell[t,n,s]) / (S_BASE * 1000)
                if n != slack:
                    mdl.add_constraint(p_buy[t,n,s] == 0)
                    mdl.add_constraint(p_sell[t,n,s] == 0)

                node_load = L_P[n][t] * load_mult_s[t]
                f_out = mdl.sum(flow[t,i,s] for i in br_out[n])
                f_in = mdl.sum(flow[t,i,s] for i in br_in[n])


                mdl.add_constraint(der + grid + (p_shed[t,n,s]/(S_BASE*1000)) - node_load == f_out - f_in)

    # ==========================================================================
    # BLOCK 3: OBJECTIVE FUNCTION (EXPECTED PROFIT)
    # ==========================================================================
    expected_profit = 0
    PRICE_MULTIPLIER = 10.0 

    for s in S_range:
        prob = scenarios[s]['prob']
        price_vec_s = scenarios[s].get('price', prof['price'])
        
        scen_net_revenue = 0
        for t in ts:
            lmp = price_vec_s[t] * PRICE_MULTIPLIER
            
            # Cálculo de Geração e Injeção no cenário s
            gen_act_s = mdl.sum((ppv[t,n,s] + pwd[t,n,s]) for n in nodes)
            inj_gross_s = mdl.sum((ppv[t,n,s] + pwd[t,n,s] + pdis[t,n,s]) for n in nodes)
            
            # Receitas e Bônus
            rev_market = inj_gross_s * (lmp * alpha)
            rev_env = gen_act_s * lmp * w_env
            rev_grid = inj_gross_s * lmp * w_grid
            rev_cap = gen_act_s * lmp * w_cap_val
            
            rev_soc = 0
            for n in nodes:
                # CORREÇÃO CRÍTICA: .strip() remove espaços "Rural " -> "Rural"
                z_raw = bus_zone_map.get(n, "Mixed")
                z = str(z_raw).strip().title() 
                
                # Pega o fator (0.0 ou 1.0)
                alloc_factor = alloc_map_cfg.get(z, 0.0)
                
                # Injeção Líquida
                loc_inj = (ppv[t,n,s] + pwd[t,n,s] + pdis[t,n,s] - pch[t,n,s])
                
                # Acumula
                rev_soc += loc_inj * lmp * w_social * alloc_factor

            scen_net_revenue += (rev_market + rev_env + rev_grid + rev_cap + rev_soc)
            scen_net_revenue -= mdl.sum(pch[t,n,s] * lmp + pdis[t,n,s] * deg_cost for n in nodes)
            
            # Penalidades
            scen_net_revenue -= mdl.sum(p_shed[t,n,s] * 1e6 for n in nodes)
            scen_net_revenue -= mdl.sum(viol[t,b['id'],s] * 1e5 for b in branches)

        expected_profit += prob * scen_net_revenue
    count_rural = sum(1 for n in nodes if bus_zone_map.get(n, "Mixed").strip().title() == "Rural")
    print(f"   [DEBUG OPTIMIZER] Nós Rurais identificados: {count_rural} de {len(nodes)}")
    mdl.maximize(expected_profit - mdl.sum(cap[n] * capex_node_day for n in nodes))
    
    # ==========================================================================
    # SOLVE & PROCESS RESULTS
    # ==========================================================================
    print("   -> Solving Stochastic Model...")
    sol = mdl.solve()

    if not sol:
        print("   ❌ Infeasible Solution.")
        return {}, [], {}, pd.DataFrame()

    print(f"   ✅ Expected Daily Profit: ${sol.objective_value:,.2f}")

   # ==========================================================================
    # 🕵️ DEBUG INTELIGENTE (MOSTRA APENAS QUEM INSTALOU)
    # ==========================================================================
    if sol:
        print("\n" + "="*80)
        print("🔍 RAIO-X FINANCEIRO (Apenas Nós Ativos)")
        print("="*80)

        # Filtra apenas nós onde o solver instalou algo (> 0.1 kW)
        active_nodes = [n for n in nodes if sol.get_value(cap[n]) > 0.1]
        
        if not active_nodes:
            print("   ❌ NENHUM INVESTIMENTO REALIZADO EM LUGAR NENHUM.")
        else:
            # Pega até 3 exemplos de zonas diferentes
            seen_zones = []
            exemplos = []
            for n in active_nodes:
                z = str(bus_zone_map.get(n, "Mixed")).strip().title()
                if z not in seen_zones:
                    exemplos.append(n)
                    seen_zones.append(z)
                if len(seen_zones) >= 3: break
                
            for node_id in exemplos:
                c_val = sol.get_value(cap[node_id])
                z_tipo = str(bus_zone_map.get(node_id, "Mixed")).strip().title()
                
                custo_real = c_val * capex_node_day
                
                # Estimativa de Receita (Reconstrução rápida)
                rec_base = 0
                rec_social = 0
                for s in S_range:
                    for t in ts:
                         inj = sol.get_value(ppv[t, node_id, s]) + sol.get_value(pwd[t, node_id, s]) # Simplificado
                         lmp = scenarios[s].get('price', prof['price'])[t] * PRICE_MULTIPLIER
                         
                         # Base
                         rec_base += inj * lmp * alpha * scenarios[s]['prob']
                         
                         # Social
                         alloc_f = alloc_map_cfg.get(z_tipo, 0.0)
                         rec_social += inj * lmp * w_social * alloc_f * scenarios[s]['prob']

                lucro_liquido = (rec_base + rec_social) - custo_real
                
                print(f"   ------------------------------------------------")
                print(f"   🏠 {z_tipo.upper()} Node: {node_id}")
                print(f"      Instalado:        {c_val:.2f} kW")
                print(f"      Custo Diário:    -${custo_real:.2f}")
                print(f"      Receita Mercado: +${rec_base:.2f}")
                print(f"      Bônus Social:    +${rec_social:.2f}  <-- SE ESTIVER ZERO, O INCENTIVO FALHOU")
                print(f"      LUCRO LÍQUIDO:    ${lucro_liquido:.2f}")

        print("="*80 + "\n")

    # ==========================================================================
    # PÓS-PROCESSAMENTO CORRIGIDO (Copie a partir daqui)
    # ==========================================================================
    decisions = {}
    total_capacity = 0.0
    
    # 1. Recupera Decisões de Investimento e Perfis de Geração
    for n in nodes:
        c = sol.get_value(cap[n])
        if c > 0.01:
            total_capacity += c
            avg_profile = []
            total_energy_node = 0.0 # <--- NOVA VARIÁVEL ACUMULADORA
            # Calcula o perfil médio de geração (Solar + Eólica) ponderado pelos cenários
            for t in ts:
                val_s = sum(scenarios[s]['prob'] * (sol.get_value(ppv[t,n,s]) + sol.get_value(pwd[t,n,s]) + sol.get_value(pdis[t,n,s])) for s in S_range)
                avg_profile.append(val_s / c if c > 0 else 0)
                total_energy_node += val_s
            
            decisions[n] = {
                'cap_kw': c, 
                'cap_solar': c, 
                'cap_wind': 0.0,
                'capex': c * costs["PV"]["CAPEX"], 
                'profile': avg_profile,
                'total_kwh': total_energy_node
            }

    # 2. Correção do Mapa: Fluxo Médio Esperado
    # (Isso conserta o mapa que estava vazio)
    peak_time = 12 if T > 12 else 0
    network_lines_result = []
    
    for b in branches:
        try:
            # Calcula a média ponderada do fluxo em todos os cenários
            avg_flow = sum(scenarios[s]['prob'] * sol.get_value(flow[peak_time, b['id'], s]) for s in S_range)
        except:
            avg_flow = 0.0
            
        network_lines_result.append({
            'from': b['from'], 
            'to': b['to'], 
            'flow_pu': avg_flow, 
            'capacity': b.get('p_max', 999)
        })

    # 3. Correção Financeira Detalhada (Isso conserta a receita zerada)
    gross_revenue_daily = 0.0 # Receita LIMPA (sem penalidades) para o Payback
    
    # Cria listas vazias para preencher
    fin_series = {k: [0.0]*T for k in ['Rev_Energy', 'Rev_Social', 'Rev_Environment', 'Rev_Grid', 'Rev_Capacity']}
    
    for t in ts:
        # Preço Médio na hora t
        avg_lmp = sum(scenarios[s]['prob'] * scenarios[s].get('price', prof['price'])[t] for s in S_range) * PRICE_MULTIPLIER
        
        # Calcula as médias esperadas de Geração e Injeção
        exp_gen = sum(scenarios[s]['prob'] * sum(sol.get_value(ppv[t,n,s]) + sol.get_value(pwd[t,n,s]) for n in nodes) for s in S_range)
        exp_inj_gross = sum(scenarios[s]['prob'] * sum(sol.get_value(ppv[t,n,s]) + sol.get_value(pwd[t,n,s]) + sol.get_value(pdis[t,n,s]) for n in nodes) for s in S_range)
        
        # Bônus Social Esperado
        exp_soc = 0.0
        for n in nodes:
            z = bus_zone_map.get(n, "Mixed")
            f_alloc = alloc_map_cfg.get(z.title(), 0.0)
            # Injeção Líquida do Nó n
            node_inj_net = sum(scenarios[s]['prob'] * (sol.get_value(ppv[t,n,s]) + sol.get_value(pwd[t,n,s]) + sol.get_value(pdis[t,n,s]) - sol.get_value(pch[t,n,s])) for s in S_range)
            exp_soc += node_inj_net * f_alloc

        # Preenche as listas para o gráfico
        fin_series['Rev_Energy'][t]      = exp_inj_gross * avg_lmp * alpha
        fin_series['Rev_Environment'][t] = exp_gen * avg_lmp * w_env
        fin_series['Rev_Grid'][t]        = exp_inj_gross * avg_lmp * w_grid
        fin_series['Rev_Capacity'][t]    = exp_gen * avg_lmp * w_cap_val
        fin_series['Rev_Social'][t]      = exp_soc * avg_lmp * w_social
        
        # Soma para o Total Diário (Receita Bruta)
        hourly_total = fin_series['Rev_Energy'][t] + fin_series['Rev_Environment'][t] + \
                       fin_series['Rev_Grid'][t] + fin_series['Rev_Capacity'][t] + fin_series['Rev_Social'][t]
        
        gross_revenue_daily += hourly_total

    # 4. Cálculo dos Custos Reais (Opex) e Lucro Líquido
    total_opex = sum(
        scenarios[s]['prob'] * sum(
            sol.get_value(pch[t,n,s]) * (scenarios[s].get('price', prof['price'])[t]*PRICE_MULTIPLIER) + 
            sol.get_value(pdis[t,n,s]) * deg_cost 
            for t in ts for n in nodes
        ) for s in S_range
    )
    
    unit_capex_real = (costs["PV"]["CAPEX"] * 1.0) + \
                      (costs["WIND"]["CAPEX"] * 1.0) + \
                      (costs["BESS"]["CAPEX_P"] * 0.5) + \
                      (costs["BESS"]["CAPEX_E"] * 2.0)

    total_capex = total_capacity * unit_capex_real
    
    # Net Diário Real (Remove a penalidade de -5 Bilhões do cálculo visual)
    net_daily_real = gross_revenue_daily - total_opex

    # 5. Montagem do Pacote de Dados Final
    dash_data = {
        'receita': net_daily_real,          # Valor corrigido (positivo)
        'custo': total_capex,
        'cap_total': total_capacity,
        'load_total': SOMA_BRUTA_EXCEL * T, 
        'gen_renovavel_bruta': sum(fin_series['Rev_Capacity']) / (avg_lmp * w_cap_val) if w_cap_val > 0 and avg_lmp > 0 else 0,
        
        'ts_data': {
            'Price': prof['price'],
            'Load': [sum(L_P[node][t] for node in nodes) * S_BASE * 1000 for t in ts],
            # Médias ponderadas para os gráficos de linha
            'PV': [sum(scenarios[s]['prob'] * sum(sol.get_value(ppv[t,n,s]) for n in nodes) for s in S_range) for t in ts],
            'Wind': [sum(scenarios[s]['prob'] * sum(sol.get_value(pwd[t,n,s]) for n in nodes) for s in S_range) for t in ts],
            'ESS_Dis': [sum(scenarios[s]['prob'] * sum(sol.get_value(pdis[t,n,s]) for n in nodes) for s in S_range) for t in ts],
            'ESS_Ch': [sum(scenarios[s]['prob'] * sum(sol.get_value(pch[t,n,s]) for n in nodes) for s in S_range) for t in ts],
            'SoC_Pct': [
            (
                sum(scenarios[s]['prob'] * sum(sol.get_value(soc[t,n,s]) for n in nodes) for s in S_range) 
                / (total_capacity * 2.0) * 100 
            ) if total_capacity > 0.01 else 0.0 
            for t in ts
        ],
        },
        
        'financial_ts': fin_series, # Aqui estão os dados que faltavam para o gráfico de pizza
        'network_map': {'coords': bus_coords, 'lines': network_lines_result} # Mapa preenchido
    }

    return decisions, prof['price'], dash_data, pd.DataFrame()