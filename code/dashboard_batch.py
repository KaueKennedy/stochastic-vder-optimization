import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
import os
import pickle
import random
from datetime import datetime
import glob
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.colors as mcolors

# Import existing modules
import data_loader
import optimizer
import simulation 
from config import CFG
import export_grid_from_dss as export_grid_from_dss
from itertools import product
import subprocess
import sys

import os
import contextlib

# --- FERRAMENTA DE SEGURANÇA ---
@contextlib.contextmanager
def preserve_cwd():
    """Garante que o diretório de trabalho volte ao original, não importa o que aconteça."""
    original_cwd = os.getcwd()
    try:
        yield
    finally:
        os.chdir(original_cwd)

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

def app():

    # ==============================================================================
    # 0. AUTO-EXTRACTION (RUNS ONCE AT STARTUP)
    # ==============================================================================
    # This ensures the Excel is always synchronized with the current Master.dss
    if 'dss_extracted' not in st.session_state:
        with st.spinner("🔄 Detecting network... Regenerating Excel from Master.dss..."):
            try:
                # Directory protection is CRITICAL here as export_grid manipulates DSS
                with preserve_cwd():
                    # Import the extraction function
                    from export_grid_from_dss import export_grid_to_excel
                    
                    # Use paths defined in CFG
                    dss_source = CFG["DSS_FILE"]
                    excel_target = CFG["EXCEL_REDE"]
                    
                    # Execute extraction
                    export_grid_to_excel(dss_source, excel_target)
                
                st.toast("✅ DSS Topology successfully updated!", icon="🏗️")
                st.session_state.dss_extracted = True
                
            except Exception as e:
                st.error(f"❌ Failed to export DSS: {e}")
                st.stop()
    # ==============================================================================
    # FUNÇÕES
    # ==============================================================================

    def plot_interactive_network(decisions, network_data, df_loads, profiles, unique_key="map_plot", plot_component=True):
        """
        Gera diagrama interativo com Tamanho baseado em Geração Total (kWh) e Tooltips customizados.
        """
        if plot_component:
            st.markdown("### 🕸️ Network Topology & Net Energy Balance")
        
        coords = network_data.get('coords', {})
        lines_data = network_data.get('lines', [])
        
        def clean_name(name):
            n = str(name).lower().strip()
            for tag in ['t_', 'load_', '_l', '_node', 'source']:
                n = n.replace(tag, '')
            return n

        # --- PASSO 0: MAPA DE DECISÕES AGREGADO ---
        mapped_decisions = {}
        if decisions:
            for d_key, d_val in decisions.items():
                k_clean = clean_name(d_key)
                if k_clean not in mapped_decisions:
                    mapped_decisions[k_clean] = {'cap_kw': 0.0, 'profile': None}
                
                mapped_decisions[k_clean]['cap_kw'] += d_val.get('cap_kw', 0.0)
                if d_val.get('profile') is not None:
                    mapped_decisions[k_clean]['profile'] = d_val['profile']

        # 1. Preparar Perfis e Totais
        prof_solar = np.array(profiles['solar'])
        prof_load = np.array(profiles['load_curve'])
        prof_wind = np.array(profiles['wind'])
        
        wind_sum = sum(prof_wind)
        solar_sum = sum(prof_solar)
        load_sum = sum(prof_load)
        
        # Mapear Carga Base
        bus_base_load = {}
        if df_loads is not None:
            for _, row in df_loads.iterrows():
                c_name = clean_name(row.get('Bus', ''))
                kw_val = float(row.get('kW', 0.0))
                bus_base_load[c_name] = bus_base_load.get(c_name, 0.0) + kw_val

        # 2. Grafo
        G = nx.Graph()
        for l in lines_data: G.add_edge(l['from'], l['to'])
        pos = coords if len(coords) > 2 else nx.kamada_kawai_layout(G)

        # 3. Desenhar Linhas
        edge_x, edge_y = [], []
        for edge in G.edges():
            if edge[0] in pos and edge[1] in pos:
                x0, y0 = pos[edge[0]]; x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None]); edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1.0, color='#95a5a6'), hoverinfo='none', mode='lines')

        # 4. Preparar Nós
        ex_x, ex_y, ex_text, ex_val, ex_size = [], [], [], [], []
        im_x, im_y, im_text = [], [], []
        audit_data = [] 
        
        # Encontrar Máxima Geração (kWh) para escalar o tamanho das bolinhas
        max_gen_kwh = 1.0
        if mapped_decisions:
            max_inst = max([d['cap_kw'] for d in mapped_decisions.values()] + [1.0])
            max_gen_kwh = max_inst * solar_sum # Estimativa máxima possível

        count_ders_found = 0

        for node in G.nodes():
            if node in pos:
                x, y = pos[node]
                c_node = clean_name(node)
                
                decision_data = mapped_decisions.get(c_node)
                base_kw = bus_base_load.get(c_node, 0.0)
                cap_inst = decision_data['cap_kw'] if decision_data else 0.0
                
                # --- CÁLCULOS DE ENERGIA (kWh) ---
                # Tenta pegar o perfil REAL (Otimizado/Despachado)
                if decision_data and decision_data.get('profile') is not None:
                    # Soma do perfil unitário real (kWh/kW)
                    gen_factor = sum(decision_data['profile'])
                else:
                    # Fallback: Perfil Potencial (Ideal)
                    gen_factor = solar_sum + wind_sum

                # Cálculo Final
                total_gen_kwh = cap_inst * gen_factor
                total_load_kwh = base_kw * load_sum
                net_energy_balance = total_gen_kwh - total_load_kwh
                
                has_der = cap_inst > 0.1
                if has_der: count_ders_found += 1
                
                # --- DEFINIÇÃO DO TIPO (NET BALANCE) ---
                if net_energy_balance > 0.001:
                    type_label = "NET EXPORTER"
                    status_icon = "📤"
                elif net_energy_balance < -0.001:
                    type_label = "NET IMPORTER"
                    status_icon = "📥"
                else:
                    type_label = "PASSIVE"
                    status_icon = "⚪"

                # --- AUDITORIA ---
                audit_data.append({
                    'Bus': node,
                    'Type': type_label,
                    'Capacity (kW)': round(cap_inst, 2),
                    'Total Gen (kWh)': round(total_gen_kwh, 1),
                    'Total Load (kWh)': round(total_load_kwh, 1),
                    'Net Balance (kWh)': round(net_energy_balance, 1)
                })

                # --- TOOLTIP CUSTOMIZADO ---
                info =  f"Bus name: {node}<br>"
                info += f"Type: {type_label}<br>"
                info += f"---------<br>"
                info += f"Installed cap: {cap_inst:.2f} kW<br>"
                info += f"Load: {base_kw:.2f} kW<br>"
                info += f"---------<br>"
                info += f"Total Gen: {total_gen_kwh:.1f} kWh<br>" 
                info += f"Total Load: {total_load_kwh:.1f} kWh<br>"
                info += f"Net Balance: {net_energy_balance:+.1f} kWh"

                # --- VISUALIZAÇÃO ---
                if has_der:
                    ex_x.append(x); ex_y.append(y)
                    ex_val.append(cap_inst) # Mantém cor baseada na capacidade instalada (intensidade)
                    
                    # TAMANHO: Baseado na Geração Total (kWh)
                    # Escala mínima 10, máxima 35
                    min_size = 1  # Tamanho da menor bolinha (geração ~0)
                    max_size = 20  # Tamanho da maior bolinha (geração máxima)
                    
                    # Cálculo proporcional
                    if max_gen_kwh > 0:
                        ratio = total_gen_kwh / max_gen_kwh
                        # Garante que o ratio não passe de 1.0 (caso haja algum outlier)
                        ratio = min(ratio, 1.0) 
                        sz = min_size + (ratio * (max_size - min_size))
                    else:
                        sz = min_size
                    
                    ex_size.append(sz)
                    ex_text.append(info)
                else:
                    im_x.append(x); im_y.append(y)
                    im_text.append(info)

        # 5. Plotagem
        # Trace PASSIVO (Cinza, tamanho fixo pequeno)
        importer_trace = go.Scatter(
            x=im_x, y=im_y, mode='markers', text=im_text, hovertemplate="%{text}<extra></extra>",
            marker=dict(color='#bdc3c7', size=6, line=dict(width=1, color='#7f8c8d')),
            name='Passive / No DER'
        )
        
        # Trace ATIVO (Com DER - Colorido por Cap, Tamanho por Energia)
        exporter_trace = go.Scatter(
            x=ex_x, y=ex_y, mode='markers', text=ex_text, hovertemplate="%{text}<extra></extra>",
            marker=dict(
                showscale=True, 
                colorscale='YlOrRd', 
                cmin=0, cmax=max([d['cap_kw'] for d in mapped_decisions.values()] + [1.0]),
                color=ex_val, 
                size=ex_size, # <--- Tamanho agora reflete kWh gerado
                line=dict(width=1, color='#2c3e50'), 
                colorbar=dict(thickness=15, title=dict(text='Capacity (kW)', side='right'), xanchor='left')
            ),
            name='Nodes with DER'
        )

        fig = go.Figure(
            data=[edge_trace, importer_trace, exporter_trace], 
            layout=go.Layout(
                title=f'Topology (Size=Gen kWh | Color=Cap kW)', 
                showlegend=True, 
                margin=dict(b=10,l=10,r=10,t=30), 
                xaxis=dict(showticklabels=False), 
                yaxis=dict(showticklabels=False)
            )
        )
        
        if plot_component:
            st.plotly_chart(fig, width="stretch", key=unique_key)
                
        return pd.DataFrame(audit_data)
    
    # ==============================================================================
    # ==============================================================================
    # --- PAGE CONFIGURATION ---
    st.set_page_config(page_title="DER Optimization Dashboard", layout="wide", page_icon="⚡")

    st.title("⚡ DER Optimization Dashboard")
    st.markdown("Interactive control of Technical, Economic, and Social parameters.")

    # ==============================================================================
    # DATA LOADING
    # ==============================================================================
    @st.cache_data
    def load_data_dynamic(t_horizon):
        CFG["T"] = t_horizon 
        
        # [CORREÇÃO 1] Protege o carregamento inicial
        with preserve_cwd():
            lines, buses, loads, prof, fig_zones = data_loader.load_inputs()
            
        return lines, buses, loads, prof, fig_zones

    # Default T
    default_t = 24

    try:
        lines, buses, loads, prof, fig_zones = load_data_dynamic(default_t)
        
        # --- CAPTURA AS HORAS DE PICO IDENTIFICADAS ---
        identified_peak_hours = prof.get('peak_hours_indices', [])
        
        zone_map = dict(zip(buses['BusName'], buses['Zone']))
        if 'SocialScore' in buses.columns:
            score_map = dict(zip(buses['BusName'], buses['SocialScore']))
        else:
            score_map = {b: 0 for b in buses['BusName']} 
    except Exception as e:
        st.error(f"Critical Error loading data: {e}")
        st.stop()

    # --- C. DEFINIÇÃO DA FUNÇÃO INTERNA ---
    def rodar_cenario_completo(arg_w_grid, arg_w_cap, arg_w_soc, arg_w_env, peak):
        nonlocal lines, buses, loads, prof, fig_zones
        with st.spinner('Running Stochastic Optimization...'):
            
            # 1. ATUALIZAÇÃO DO CONFIG COM DADOS DA TELA
            CFG["STOCHASTIC"]["NUM_SCENARIOS"] = in_scenarios
            CFG["STOCHASTIC"]["SIGMA_LOAD"] = in_sigma_load
            CFG["STOCHASTIC"]["SIGMA_RENEWABLE"] = in_sigma_ren
            CFG["STOCHASTIC"]["MIP_GAP"] = in_mipgap

            # 1. RELOAD DATA (Se mudou o Horizonte T)
            if int(cfg_t) != default_t:
                lines, buses, loads, prof, fig_zones = load_data_dynamic(int(cfg_t))
                identified_peak_hours = prof.get('peak_hours_indices', [])
            else:
                identified_peak_hours = peak

            # 2. INJECT CONFIG (Global CFG)
            CFG["T"] = int(cfg_t)
            CFG["REMUNERATION"]["PEAK_HOURS"] = identified_peak_hours
          
            # --- CÁLCULO DOS FATORES DE AMORTIZAÇÃO (Diário) ---
            # O otimizador precisa saber quanto do custo total é alocado por dia
            # Fator = 1 / (Anos * 365)
            f_pv   = 1.0 / (life_pv * 365.0)
            f_wind = 1.0 / (life_wind * 365.0)
            f_bess = 1.0 / (life_bess * 365.0)

            # --- INJECT DETAILED COSTS INTO GLOBAL CFG ---
            CFG["COSTS"] = {
                "PV": {
                    "CAPEX": float(capex_pv),   
                    "OM": float(om_pv),   
                    "AMORT": f_pv,  # Passamos o fator de tempo
                    "LIFE": float(life_pv)
                },
                "WIND": {
                    "CAPEX": float(capex_wind), 
                    "OM": float(om_wind), 
                    "AMORT": f_wind,
                    "LIFE": float(life_wind)
                },
                "BESS": {
                    "CAPEX_P": float(capex_bess_p), 
                    "CAPEX_E": float(capex_bess_e), 
                    "OM": float(om_bess), 
                    "AMORT": f_bess,         
                    "DEG_COST": deg_cost_kwh,
                    "LIFE": float(life_bess)
                },
                "ECON": {
                    "WACC": float(interest_rate) # Taxa de Juros
                }
            }

            # Atualiza Pesos e Alocação
            tot_alloc = alloc_rur + alloc_mix + alloc_urb
            if tot_alloc == 0: tot_alloc = 1.0
            
            CFG["REMUNERATION"]["SOCIAL_ALLOCATION"] = {
                "Rural": alloc_rur / tot_alloc,
                "Mixed": alloc_mix / tot_alloc,
                "Urban": alloc_urb / tot_alloc
            }

            CFG["REMUNERATION"]["W_GRID"]     = {"val": float(arg_w_grid)}
            CFG["REMUNERATION"]["W_CAPACITY"] = {"val": float(arg_w_cap)}
            CFG["REMUNERATION"]["W_SOCIAL"]   = {"val": float(arg_w_soc)}
            CFG["REMUNERATION"]["W_ENV"]      = {"val": float(arg_w_env)}
            CFG["REMUNERATION"]["ALPHA"]    = float(cfg_alpha)

            # Atualiza Limites Físicos
            CFG["MAX_PENETRATION"] = cfg_max_pen
            CFG["V_MIN"] = cfg_v_min; CFG["V_MAX"] = cfg_v_max
            CFG["OPT_V_MIN"] = cfg_opt_v_min; CFG["OPT_V_MAX"] = cfg_opt_v_max
            CFG["ZONAL_LIMITS"]["Urban"] = {"min": lim_z_urb[0], "max": lim_z_urb[1]}
            CFG["ZONAL_LIMITS"]["Mixed"] = {"min": lim_z_mix[0], "max": lim_z_mix[1]}
            CFG["ZONAL_LIMITS"]["Rural"] = {"min": lim_z_rur[0], "max": lim_z_rur[1]}
            
            # Penalidade
            CFG["PENALTY"] = cfg_penalty

            scenarios_data = data_loader.generate_stochastic_scenarios(prof, CFG["STOCHASTIC"])

            # =========================================================
            # 3. CARREGAMENTO EXCEL
            # =========================================================
            excel_file = EXCEL_REDE
            
            try:
                xls = pd.ExcelFile(excel_file)
                lines_df = pd.read_excel(xls, "Lines")
                xfmr_df  = pd.read_excel(xls, "Transformers")
                buses_df = pd.read_excel(xls, "Buses")
                loads_df = pd.read_excel(xls, "Loads")
            except Exception as e:
                st.error(f"❌ Error loading Excel: {e}")
                st.stop()

            # =========================================================
            # 4. CHAMAR OTIMIZADOR (COM PROTEÇÃO)
            # =========================================================
            start_t = time.time()
            
            # Envolvemos a chamada na trava de segurança
            with preserve_cwd():
                # AQUI NASCE A VARIÁVEL 'decisions'
                decisions, price_vec, dash_data, _ = optimizer.run_optimization(
                    lines, buses, loads, prof, 
                    transformers=None, 
                    scenarios=scenarios_data
                )

            # =========================================================
            # 5. VERIFICAR RESULTADOS (DENTRO DO IF RUN_BTN)
            # =========================================================
            
            if decisions is None:
                st.error("❌ Critical Error: The Solver could not find a feasible solution (Infeasible).")
                st.warning("Try increasing capacity limits in the 'Limits' tab or relaxing constraints.")
                st.session_state.sim_results = None
                
            else:
                # Caso de Sucesso (Mesmo que vazio/sem investimento)
                if not decisions:
                    st.warning("⚠️ Optimization successful, but NO investment was made (Cost > Revenue).")
                    st.info(f"💵 Objective Function (Net Value): ${dash_data.get('receita', 0):,.2f}")
                else:
                    st.success(f"✅ Optimization Completed! Net Value: ${dash_data.get('receita', 0):,.2f}")

                # Plot da Rede Interativa
                if 'network_map' in dash_data:
                    # Este é o gráfico que já existe na tela principal
                    plot_interactive_network(
                        decisions, 
                        dash_data['network_map'], 
                        loads, 
                        prof, 
                        unique_key=f"map_main_tab_{time.time()}",
                        plot_component=True
                    )
                    #if df_map_audit is not None and not df_map_audit.empty:
                    #    st.dataframe(df_map_audit, width="stretch")

                    # [MUDANÇA 2] Adicione estas linhas logo abaixo:
                    st.markdown("---")
                    st.markdown("### 🗺️ Network Zones (Geographic Distribution)")
                    # Recupera o gráfico gerado durante o load_inputs
                    lines, buses, loads, prof, fig_zones = data_loader.load_inputs()
                    st.plotly_chart(fig_zones, width="stretch", key=f"zones_{time.time()}")

                # 6. DIGITAL TWIN (Física Real)
                with preserve_cwd():
                    full_data_dss = simulation.run_digital_twin(decisions)
                
                elapsed = time.time() - start_t
                st.success(f"⏱️ Converged in {elapsed:.2f}s")

        # ==========================================================
        # CONFIGURAÇÃO DAS VARIÁVEIS (DEFINIÇÃO QUE FALTAVA)
        # ==========================================================
        
        # 1. Define a Data Inicial
        start_date = "2024-01-01 00:00:00"

        # 2. Cria os controles para definir 'freq_code'
        c_ctrl1, c_ctrl2 = st.columns([2, 1])

        # 4. Cria as colunas 'c_bot1' e 'col_map'
        c_bot1, col_map = st.columns([1, 1])

        with col_map:

                # KPIs Rápidos
                rev_total = dash_data['receita']
                cap_total = dash_data['cap_total']
                idx_perf = rev_total / cap_total if cap_total > 0 else 0
                
                # Estrutura Final
                result_package = {
                    'decisions': decisions, 
                    'dash_data': dash_data, 
                    'df_v_real': full_data_dss['voltages'], 
                    'kpis': {
                        'perf': idx_perf, 
                        'cap': cap_total,
                        'pen': (dash_data['gen_renovavel_bruta'] / dash_data['load_total'] * 100) if dash_data['load_total'] > 0 else 0,
                        'res': 0, # placeholders se quiser recalcular
                        'soc': 0
                    },
                    'config': CFG.copy() # Salva a config usada para referência futura
                }
                
                st.session_state.sim_results = result_package
                st.session_state.loaded_from_history = False

                # 8. LOG DE HISTÓRICO
                timestamp_id = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"sim_{timestamp_id}.pkl"
                with open(os.path.join(HISTORY_DIR, filename), 'wb') as f: 
                    pickle.dump(result_package, f)
                    
                # CSV Log
                log_entry = {
                    'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'File': filename,
                    'Performance_Index': idx_perf,
                    'Total_Capacity_kW': cap_total
                }
                df_new = pd.DataFrame([log_entry])
                if os.path.exists(history_csv):
                    try: pd.read_csv(history_csv); df_new.to_csv(history_csv, mode='a', header=False, index=False)
                    except: df_new.to_csv(history_csv, mode='w', header=True, index=False)
                else: df_new.to_csv(history_csv, mode='w', header=True, index=False)

                # ==============================================================================
                # NOVO: EXPORTAÇÃO DE DADOS HORÁRIOS PARA TXT (KPIs)
                # ==============================================================================
                
                try:
                    # 1. Recupera Séries Temporais do Sistema (Já calculadas)
                    tsd = dash_data['ts_data']
                    fin = dash_data['financial_ts']
                    
                    # 2. Calcula Séries Específicas da Zona Rural (Para KPI de Autossuficiência)
                    # Otimização: Fazemos isso aqui para evitar reprocessar tudo no Excel
                    rural_load_ts = np.zeros(CFG['T'])
                    rural_der_gen_ts = np.zeros(CFG['T'])
                    
                    # A. Soma Carga Rural
                    if 'nodal_loads' in dash_data:
                        for bus, profile_pu in dash_data['nodal_loads'].items():
                            if zone_map.get(bus, 'Mixed') == 'Rural':
                                # Converte p.u. -> kW
                                rural_load_ts += profile_pu * CFG["S_BASE"] * 1000
                    
                    # B. Soma Geração DER Rural
                    if decisions:
                        for bus, dec in decisions.items():
                            if zone_map.get(bus, 'Mixed') == 'Rural':
                                rural_der_gen_ts += np.array(dec['profile'])

                    # 3. Monta o DataFrame Consolidado
                    # Calcula Injeção Líquida DER = (PV + Wind + Discharge - Charge)
                    net_der = np.array(tsd['PV']) + np.array(tsd['Wind']) + \
                            np.array(tsd['ESS_Dis']) - np.array(tsd['ESS_Ch'])
                    
                    df_export = pd.DataFrame({
                        'Hora': range(CFG['T']),
                        'Preco_LMP_($/kWh)': tsd['Price'],
                        # Dados Físicos Totais
                        'Carga_Total_(kW)': tsd['Load'],
                        'PV_Total_(kW)': tsd['PV'],
                        'Wind_Total_(kW)': tsd['Wind'],
                        'Bess_Discharge_(kW)': tsd['ESS_Dis'],
                        'Bess_Charge_(kW)': tsd['ESS_Ch'],
                        'DER_Net_Injection_(kW)': net_der,
                        # Dados Financeiros (Receitas)
                        'Rec_Energia_($)': fin['Energy'],
                        'Rec_Ambiental_($)': fin['Environment'],
                        'Rec_Rede_($)': fin['GridSupport'],
                        'Rec_Social_($)': fin['SocialEquity'],
                        'Rec_Capacidade_($)': fin['Capacity'],
                        # Dados Rurais (Para KPI 5)
                        'Carga_Rural_(kW)': rural_load_ts,
                        'Geracao_Rural_(kW)': rural_der_gen_ts,
                        'Pico_Identificado': [1 if t in identified_peak_hours else 0 for t in range(CFG['T'])] # Coluna Extra para verificação
                    })

                except Exception as e:
                    None

        # ==============================================================================
        # VISUALIZATION
        # ==============================================================================
        df_history_full = pd.read_csv(history_csv) if os.path.exists(history_csv) else pd.DataFrame()
        if not df_history_full.empty and 'File' not in df_history_full.columns: df_history_full['File'] = None

        c_3d, c_select = st.columns([3, 1])



        if not df_history_full.empty:
            df_history_full['Label'] = df_history_full.apply(lambda x: f"Run {x.name} | {str(x['Timestamp'])[5:-3]} (Perf: ${x['Performance_Index']:.2f})", axis=1)
        else: df_history_full['Label'] = []

        selected_run_row = None

            # --- DETAILED ANALYSIS ---
        if st.session_state.sim_results:
            res = st.session_state.sim_results
            dash_data = res['dash_data']
            
            # Se carregou de Batch, df_v_real pode ser None. Tratar isso.
            df_v_real = res.get('df_v_real') 
            
            decisions = res['decisions']
            kpis = res['kpis']
            saved_config = res.get('config', None) # Tenta recuperar a config salva

            fin_ts = dash_data['financial_ts']

            st.markdown("---")
            
            # Abas de Análise
            tab_res, tab_cfg, tab_audit = st.tabs(["📊 Results & Charts", "📜 Scenario Config", "🔍 Data Audit"])
            
            df_map_audit = None

            # === ABA DE CONFIGURAÇÃO (NOVO) ===
            with tab_cfg:
                if saved_config:
                    st.info("Estas são as configurações EXATAS usadas nesta simulação.")
                    
                    c_cfg1, c_cfg2 = st.columns(2)
                    with c_cfg1:
                        st.subheader("💰 Costs & Economics")
                        st.json(saved_config.get("COSTS", "Not Found"))
                        st.write(f"**Alpha:** {saved_config['REMUNERATION'].get('ALPHA', '?')}")
                        
                    with c_cfg2:
                        st.subheader("⚖️ Weights (Incentives)")
                        rem = saved_config.get("REMUNERATION", {})
                        st.write(f"**Social:** {rem.get('W_SOCIAL', {}).get('val', 0):.2f}")
                        st.write(f"**Env:** {rem.get('W_ENV', {}).get('val', 0):.2f}")
                        st.write(f"**Grid:** {rem.get('W_GRID', {}).get('val', 0):.2f}")
                        st.write(f"**Capacity:** {rem.get('W_CAPACITY', {}).get('val', 0):.2f}")
                        
                    with st.expander("Ver Configuração Completa (JSON)"):
                        st.json(saved_config)
                else:
                    st.warning("⚠️ Este arquivo de resultado é antigo e não contém o registro da configuração.")
                pass

            with tab_res:
                # ==========================================================
                # 0. SEGURANÇA DE DADOS
                # ==========================================================
                try: buses_safe = buses
                except NameError:
                    try: buses_safe = buses
                    except NameError: _, buses_safe, _, _ = data_loader.load_inputs()

                # ==========================================================
                # 1. KPIs GERAIS (Topo)
                # ==========================================================
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Capacity", f"{dash_data['cap_total']:,.0f} kW", "Installed")
                c2.metric("Daily Revenue", f"${dash_data.get('receita', 0):,.2f}", "USD/Day")
                c3.metric("CAPEX", f"${dash_data.get('custo', 0):,.0f}", "One-time Cost")
                rev = dash_data.get('receita', 0)
                pb = dash_data.get('custo', 0) / (rev * 365) if rev > 0 else 0
                c4.metric("Est. Payback", f"{pb:.1f} Years", "ROI")

                st.markdown("---")

                # Configurações de Visualização
                c_ctrl1, c_ctrl2 = st.columns([3, 1])
                with c_ctrl1: st.caption("Analysis Horizon: 24h")
                with c_ctrl2: 
                    view_mode = st.radio("Granularity:", ["Hourly", "Daily"], horizontal=True, key=f"view_rad_final_{time.time()}")
                
                if view_mode == "Daily": freq_code = 'D'
                else: freq_code = 'H'
                start_date = "2024-01-01 00:00:00"

                # ==========================================================
                # LINHA 1: DESPACHO OPERACIONAL (LARGURA TOTAL)
                # ==========================================================
                st.subheader("1. Operational Dispatch (Generation vs Load)")
                
                ts = dash_data['ts_data']
                df_disp = pd.DataFrame({
                    'PV': ts['PV'], 'Wind': ts['Wind'], 'Discharge': ts['ESS_Dis'],
                    'Charge': [-x for x in ts['ESS_Ch']], 'Load': ts['Load'], 
                    'SoC': ts['SoC_Pct'] # Mantemos o valor em % (0-100)
                })
                
                # Grid = Carga + CargaBateria - GeraçãoLocal
                df_disp['Grid'] = [max(0, l + c - (p+w+d)) for l,c,p,w,d in zip(ts['Load'], ts['ESS_Ch'], ts['PV'], ts['Wind'], ts['ESS_Dis'])]
                df_disp.index = pd.date_range(start=start_date, periods=len(df_disp), freq='h')
                
                # LÓGICA DE AGRUPAMENTO (CORRIGIDA PARA SOC)
                if freq_code != 'H': 
                    # Potências: Soma
                    df_plot = df_disp.drop(columns=['SoC']).resample(freq_code).sum()
                    # SoC: Média (não faz sentido somar %)
                    df_plot['SoC'] = df_disp['SoC'].resample(freq_code).mean()
                else: 
                    df_plot = df_disp

                fig_d = go.Figure()
                colors_disp = {'PV':'#f1c40f', 'Wind':'#2ecc71', 'Discharge':'#e67e22', 'Grid':'#95a5a6', 'Charge':'#c0392b'}
                
                # 1. Plot das Barras (Potência kW - Eixo Esquerdo)
                for col, color in colors_disp.items():
                    if col in df_plot.columns:
                        fig_d.add_trace(go.Bar(x=df_plot.index, y=df_plot[col], name=col, marker_color=color, hovertemplate="%{y:,.1f} kW"))
                
                # 2. Plot da Linha de Carga (Potência kW - Eixo Esquerdo)
                fig_d.add_trace(go.Scatter(x=df_plot.index, y=df_plot['Load'], name='Load', line=dict(color='black', width=3)))
                
                # 3. Plot do SoC (Porcentagem - Eixo Direito)
                # Usamos yaxis='y2' para criar a escala 0-100 na direita
                if 'SoC' in df_plot.columns:
                    fig_d.add_trace(go.Scatter(
                        x=df_plot.index, 
                        y=df_plot['SoC'], 
                        name='Battery SoC', 
                        line=dict(color='#8e44ad', width=3, dash='dot'), # Roxo tracejado
                        yaxis='y2', # Manda para o eixo secundário
                        hovertemplate="SoC: %{y:.1f}%" # Etiqueta exigida em %
                    ))

                # Configuração do Layout com Eixo Duplo
                fig_d.update_layout(
                    barmode='relative', 
                    height=400, 
                    margin=dict(l=0,r=0,t=10,b=0), 
                    legend=dict(orientation="h", y=1.1, x=0), 
                    hovermode="x unified",
                    yaxis=dict(title="Power (kW)"), # Eixo Esquerdo
                    yaxis2=dict(
                        title="State of Charge (%)", # Eixo Direito
                        overlaying="y",
                        side="right",
                        range=[0, 110], # Fixa escala 0 a 100%
                        showgrid=False
                    )
                )
                st.plotly_chart(fig_d, width="stretch", key=f"chart_dispatch_full_{time.time()}")

                # ==========================================================
                # LINHA 2: PERFIL DE TENSÃO (CORRIGIDO)
                # ==========================================================
                st.subheader("2. Voltage Profile (p.u.)")
                
                if df_v_real is not None:
                    # 1. Limpeza: Remove a coluna 'Hora' se existir e substitui 0 por NaN
                    df_v_clean = df_v_real.drop(columns=['Hora', 'Hour'], errors='ignore').replace(0, np.nan)
                    
                    # 2. Cálculo dos Extremos Reais (Linha a Linha = Hora a Hora)
                    # axis=1 pega o max/min entre todas as colunas (barras) para aquela hora
                    max_v_curve = df_v_clean.max(axis=1)
                    min_v_curve = df_v_clean.min(axis=1)
                    
                    # Eixo X temporal
                    x_axis_v = pd.date_range(start=start_date, periods=len(df_v_clean), freq='h')
                    
                    fig_v = go.Figure()
                    
                    # Área Sombreada (Range Real)
                    fig_v.add_trace(go.Scatter(
                        x=list(x_axis_v) + list(x_axis_v)[::-1],
                        y=list(max_v_curve) + list(min_v_curve)[::-1],
                        fill='toself', fillcolor='rgba(100, 149, 237, 0.2)', 
                        line=dict(color='rgba(255,255,255,0)'), name='System Range', hoverinfo='skip'
                    ))
                    
                    # Linhas de Máximo e Mínimo Reais
                    fig_v.add_trace(go.Scatter(x=x_axis_v, y=max_v_curve, name='Real Max V', line=dict(color='#e74c3c', width=2)))
                    fig_v.add_trace(go.Scatter(x=x_axis_v, y=min_v_curve, name='Real Min V', line=dict(color='#2980b9', width=2)))
                    
                    # 3. Limites Estipulados (CFG - Dashboard)
                    # Se não tiver CFG salvo, usa padrão
                    lim_min = CFG.get("V_MIN", 0.93)
                    lim_max = CFG.get("V_MAX", 1.05)
                    
                    fig_v.add_hline(y=lim_max, line_dash="dash", line_color="black", annotation_text=f"Limit Max ({lim_max} pu)")
                    fig_v.add_hline(y=lim_min, line_dash="dash", line_color="black", annotation_text=f"Limit Min ({lim_min} pu)")
                    
                    # Ajuste de Eixo Y para não ficar começando em 0 (foca entre 0.8 e 1.1)
                    y_lower = min(lim_min, min_v_curve.min()) - 0.02
                    y_upper = max(lim_max, max_v_curve.max()) + 0.02
                    
                    fig_v.update_layout(height=350, margin=dict(l=0,r=0,t=20,b=0), 
                                        yaxis=dict(range=[y_lower, y_upper], title="Voltage (p.u.)"),
                                        hovermode="x unified")
                    st.plotly_chart(fig_v, width="stretch", key=f"chart_voltage_full_{time.time()}")
                    
                else:
                    st.warning("⚠️ Voltage data unavailable. Please run the simulation correctly to generate data.")

                # ==========================================================
                # LINHA 3: RENDA (ÁREA + PIZZA) - VERSÃO ROBUSTA
                # ==========================================================
                st.subheader("3. Economic Performance (By Incentive Factor)")
                
                c_rev_area, c_rev_pie = st.columns([3, 1]) 
                
                fin_ts = dash_data.get('financial_ts', [])
                
                if fin_ts:
                    df_fin = pd.DataFrame(fin_ts)
                    
                    # 1. Definição das colunas de Fatores (Nomes exatos do Optimizer)
                    cols_factors = ['Rev_Energy', 'Rev_Social', 'Rev_Environment', 'Rev_Grid', 'Rev_Capacity']
                    
                    # 2. Garante numérico e preenche zeros
                    for c in cols_factors: 
                        if c not in df_fin.columns: df_fin[c] = 0.0
                    
                    df_fin = df_fin.fillna(0.0)
                    
                    # 3. Índice Temporal
                    df_fin.index = pd.date_range(start=start_date, periods=len(df_fin), freq='h')
                    
                    # 4. Agrupamento (Resample) - Cria df_plot_fin usado em AMBOS os gráficos
                    if freq_code != 'H': 
                        df_plot_fin = df_fin[cols_factors].resample(freq_code).sum()
                    else: 
                        df_plot_fin = df_fin[cols_factors]

                    # Paleta de Cores
                    color_map = {
                        'Rev_Energy': '#34495e',       # Azul Escuro
                        'Rev_Social': '#9b59b6',       # Roxo
                        'Rev_Environment': '#2ecc71',  # Verde
                        'Rev_Grid': '#e67e22',         # Laranja
                        'Rev_Capacity': '#95a5a6'      # Cinza
                    }
                    
                    # --- GRÁFICO DE ÁREA (Esquerda) ---
                    with c_rev_area:
                        fig_r = px.area(df_plot_fin, x=df_plot_fin.index, y=cols_factors, 
                                        labels={'value': 'Revenue ($)', 'variable': 'Incentive'},
                                        color_discrete_map=color_map)
                        
                        # CORREÇÃO: Etiqueta explícita com nome da série e valor com 2 casas decimais
                        fig_r.update_traces(hovertemplate="<b>%{fullData.name}</b>: %{y:,.2f} $<extra></extra>") 
                        
                        fig_r.update_layout(height=400, margin=dict(l=0,r=0,t=10,b=0), 
                                            legend=dict(orientation="h", y=1.1), hovermode="x unified")
                        st.plotly_chart(fig_r, width="stretch", key=f"chart_revenue_area_{time.time()}")

                    # --- GRÁFICO DE PIZZA (Direita) ---
                    with c_rev_pie:
                        # Usa a SOMA do df_plot_fin (garante consistência total com a área)
                        sums = df_plot_fin[cols_factors].sum().to_dict()
                        
                        # CORREÇÃO: Filtro mínimo (qualquer valor positivo aparece)
                        sums_clean = {k: v for k, v in sums.items() if v > 0.000001}
                        
                        if sums_clean:
                            # Limpa nomes (Remove 'Rev_') para a legenda da pizza
                            clean_lbls = {k: k.replace('Rev_', '') for k in sums_clean.keys()}
                            
                            fig_p = px.pie(
                                values=list(sums_clean.values()), 
                                names=[clean_lbls[k] for k in sums_clean.keys()], 
                                hole=0.5, 
                                color_discrete_sequence=[color_map[k] for k in sums_clean.keys()]
                            )
                            
                            fig_p.update_layout(height=400, margin=dict(l=0,r=0,t=30,b=0), 
                                                showlegend=False, 
                                                annotations=[dict(text='TOTAL', x=0.5, y=0.5, font_size=12, showarrow=False)])
                            
                            # Tooltip detalhado na pizza
                            fig_p.update_traces(hovertemplate="<b>%{label}</b><br>%{value:,.2f} $<br>(%{percent})")
                            
                            st.plotly_chart(fig_p, width="stretch", key=f"chart_revenue_pie_{time.time()}")
                        else:
                            st.info("No Revenue Generated")
                else:
                    st.info("No financial data available.")

                # ==========================================================
                # LINHA 4: CAPACIDADE POR ZONA (LARGURA TOTAL)
                # ==========================================================
                st.subheader("4. Capacity Allocation by Zone")
                
                # Mapa de Zonas
                zone_map = dict(zip(buses_safe['BusName'], buses_safe['Zone']))
                zone_energy = {'Urban': 0.0, 'Mixed': 0.0, 'Rural': 0.0}
                
                for bus, d in decisions.items():
                    z = zone_map.get(bus, 'Mixed')
                    kwh_val = d.get('total_kwh', 0.0)
                    if z in zone_energy: zone_energy[z] += kwh_val
                    else: zone_energy['Mixed'] += kwh_val
                
                df_z = pd.DataFrame(list(zone_energy.items()), columns=['Zone', 'kWh'])
                tot = df_z['kWh'].sum()
                df_z['Pct'] = (df_z['kWh']/tot*100) if tot > 0 else 0
                
                fig_z = px.bar(df_z, x='Zone', y='kWh', text='Pct', color='Zone', 
                            color_discrete_map={'Urban':'#3498db', 'Mixed':'#f1c40f', 'Rural':'#2ecc71'})
                
                fig_z.update_traces(texttemplate='%{text:.1f}%', textposition='outside', hovertemplate="%{y:,.1f} kWh")
                fig_z.update_layout(height=400, margin=dict(l=0,r=0,t=10,b=0))
                st.plotly_chart(fig_z, width="stretch", key=f"chart_zonal_final_{time.time()}")

                # ==============================================================================
                # 5. EXPORTAÇÃO MASTER (BANCO DE DADOS EM CSV)
                # ==============================================================================
                st.markdown("---")
                st.subheader("💾 Master Data Logging (All Simulations Combined)")
                
                # Identificadores da Rodada Atual
                sim_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                sim_id = datetime.now().strftime("%Y%m%d_%H%M%S") # ID curto
                
                # Recupera Parâmetros Chave (Para você filtrar nos gráficos depois)
                rem = CFG["REMUNERATION"]
                meta_cols = {
                    'Sim_ID': sim_id,
                    'Timestamp': sim_timestamp,
                    'Max_Penetration': CFG.get("MAX_PENETRATION", 0),
                    'Weight_Social': rem['W_SOCIAL']['val'],
                    'Weight_Grid': rem['W_GRID']['val'],
                    'Weight_Env': rem['W_ENV']['val'],
                    'Weight_Cap': rem['W_CAPACITY']['val'],
                    'WACC': CFG['COSTS']['ECON']['WACC'],
                    'Total_Revenue': dash_data['receita'],
                    'Total_Capex': dash_data['custo']
                }

                try:
                    # --- ARQUIVO 1: MASTER TIME SERIES (Hora a Hora) ---
                    # Contém: Carga, Geração, Bateria, Finanças (24 linhas por simulação)
                    tsd = dash_data['ts_data']
                    fin_ts = dash_data['financial_ts']
                    
                    net_der = np.array(tsd['PV']) + np.array(tsd['Wind']) + np.array(tsd['ESS_Dis']) - np.array(tsd['ESS_Ch'])
                    
                    df_ts = pd.DataFrame({
                        'Hora': range(CFG['T']),
                        'Preco_LMP': tsd['Price'],
                        'Carga_Total': tsd['Load'],
                        'PV_Total': tsd['PV'],
                        'Wind_Total': tsd['Wind'],
                        'Bess_Discharge': tsd['ESS_Dis'],
                        'Bess_Charge': tsd['ESS_Ch'],
                        'Bess_SoC_Pct': tsd['SoC_Pct'],
                        'DER_Injecao_Liq': net_der,
                        # Finanças
                        'Rev_Energia': fin_ts.get('Rev_Energy', [0]*CFG['T']),
                        'Rev_Social': fin_ts.get('Rev_Social', [0]*CFG['T']),
                        'Rev_Grid': fin_ts.get('Rev_Grid', [0]*CFG['T'])
                    })
                    
                    # Adiciona Metadados (repete em todas as linhas da rodada)
                    for k, v in meta_cols.items():
                        df_ts.insert(0, k, v) # Insere no começo

                    # Salva (Append Mode)
                    file_ts = os.path.join(CFG["OUTPUT_DIR"], "MASTER_Hourly_Results.csv")
                    header_ts = not os.path.exists(file_ts) # Escreve cabeçalho só se arquivo não existir
                    df_ts.to_csv(file_ts, mode='a', header=header_ts, index=False, float_format='%.4f')


                    # --- ARQUIVO 2: MASTER BUS DATA (Barra a Barra) ---
                    # Contém: Capacidade instalada, Geração Total, Tipo (N linhas por simulação)
                    # Reusa a função do mapa para garantir consistência
                    df_bus = plot_interactive_network(
                        decisions, dash_data['network_map'], loads, prof, 
                        plot_component=False # Só calcula
                    )
                    
                    if df_bus is not None and not df_bus.empty:
                        # Adiciona Metadados
                        for k, v in meta_cols.items():
                            df_bus.insert(0, k, v)
                            
                        file_bus = os.path.join(CFG["OUTPUT_DIR"], "MASTER_Bus_Results.csv")
                        header_bus = not os.path.exists(file_bus)
                        df_bus.to_csv(file_bus, mode='a', header=header_bus, index=False, float_format='%.4f')


                    # --- ARQUIVO 3: MASTER VOLTAGES (Forense) ---
                    # Contém: Tensões de todas as barras (24 linhas por simulação, muitas colunas)
                    if df_v_real is not None:
                        df_volt = df_v_real.copy()
                        # Adiciona Metadados
                        for k, v in meta_cols.items():
                            df_volt.insert(0, k, v)
                            
                        file_volt = os.path.join(CFG["OUTPUT_DIR"], "MASTER_Voltage_Log.csv")
                        header_volt = not os.path.exists(file_volt)
                        df_volt.to_csv(file_volt, mode='a', header=header_volt, index=False, float_format='%.4f')

                    # --- Feedback Sucesso ---
                    st.success(f"✅ Data Appended to Master Files (ID: {sim_id})")
                    c_d1, c_d2, c_d3 = st.columns(3)
                    c_d1.info(f"📄 Hourly Rows: {len(df_ts)} added")
                    if df_bus is not None: c_d2.info(f"📄 Bus Rows: {len(df_bus)} added")
                    c_d3.caption(f"📁 Location: {CFG['OUTPUT_DIR']}")

                except Exception as e:
                    st.error(f"Error appending data: {e}")
                    
                    c_exp1, c_exp2 = st.columns([1, 2])
                    with c_exp1:
                        st.success(f"✅ Data Saved: `{sim_id}`")
                        st.download_button("Download .csv", df_export.to_csv(sep='\t', index=False).encode('utf-8'), 
                                            file_name=f"{sim_id}.csv", mime="text/plain")
                    # ==========================================================
                    # D. LOG DE HISTÓRICO PARA O GRÁFICO 3D
                    # ==========================================================
                    arquivo_historico_3d = os.path.join(CFG["OUTPUT_DIR"], "Global_History_Log.csv")
                    
                    # Recupera Pesos Usados
                    rem = CFG["REMUNERATION"]
                    w_soc = rem['W_SOCIAL']['val']
                    w_env = rem['W_ENV']['val']
                    w_grid = rem['W_GRID']['val']
                    w_cap = rem['W_CAPACITY']['val']
                    
                    # Calcula Eixos do 3D
                    axis_x_social = w_soc + w_env   # Foco Socioambiental
                    axis_y_tech = w_grid + w_cap    # Foco Técnico
                    
                    # Eixo Z: Lucratividade (Receita - Capex) ou ROI
                    profit = dash_data['receita'] * 365 * 10 - dash_data['custo'] # Estimativa 10 anos
                    roi_index = profit / dash_data['custo'] if dash_data['custo'] > 0 else 0
                    
                    new_entry = pd.DataFrame([{
                        'Sim_ID': sim_id,
                        'Timestamp': datetime.now(),
                        'Axis_X_SocialEnv': axis_x_social,
                        'Axis_Y_Tech': axis_y_tech,
                        'Axis_Z_Profit': profit,
                        'ROI_Index': roi_index,
                        'Total_Revenue': dash_data['receita'],
                        'Total_Capex': dash_data['custo'],
                        'Total_Capacity': dash_data['cap_total'],
                        'Weights_Used': f"S:{w_soc}|E:{w_env}|G:{w_grid}|C:{w_cap}"
                    }])
                    
                    # Append no CSV de histórico
                    if os.path.exists(arquivo_historico_3d):
                        new_entry.to_csv(arquivo_historico_3d, mode='a', header=False, index=False)
                    else:
                        new_entry.to_csv(arquivo_historico_3d, mode='w', header=True, index=False)
                    
                    # ==========================================================
                    # E. PLOTAGEM DO GRÁFICO 3D (POSIÇÃO DA SIMULAÇÃO)
                    # ==========================================================
                    with c_exp2:
                        st.subheader("🧊 Profitability Landscape (3D History)")
                        
                        # Lê o histórico completo
                        df_hist = pd.read_csv(arquivo_historico_3d)
                        
                        fig_3d = px.scatter_3d(
                            df_hist, 
                            x='Axis_X_SocialEnv', 
                            y='Axis_Y_Tech', 
                            z='Axis_Z_Profit',
                            color='ROI_Index',
                            size='Total_Capacity',
                            hover_data=['Sim_ID', 'Weights_Used'],
                            color_continuous_scale='Viridis',
                            title="Optimization Trajectory (Your run is marked 🔴)"
                        )
                        
                        # Adiciona um ponto VERMELHO GRANDE representando a simulação ATUAL
                        fig_3d.add_trace(go.Scatter3d(
                            x=[axis_x_social], 
                            y=[axis_y_tech], 
                            z=[profit],
                            mode='markers+text',
                            marker=dict(size=15, color='red', symbol='diamond', opacity=1.0),
                            text=["CURRENT RUN"],
                            textposition="top center",
                            name='Current Run'
                        ))
                        
                        fig_3d.update_layout(
                            scene=dict(
                                xaxis_title='Social + Env Focus (X)',
                                yaxis_title='Grid + Cap Focus (Y)',
                                zaxis_title='10y Net Profit ($) (Z)'
                            ),
                            margin=dict(l=0, r=0, b=0, t=30),
                            height=400
                        )
                        
                        st.plotly_chart(fig_3d, width="stretch", key=f"chart_3d_history_{time.time()}")
                        st.info(f"📍 Current Position ID: **{sim_id}**")

                    

                except Exception as e:
                    st.error(f"Erro ao processar exportação/histórico: {e}")

            with tab_audit:
                st.header("🔍 Data Audit")
                st.info("This section provides detailed data tables used in the various charts for verification purposes.")

                # 1. Auditoria do Mapa
                st.subheader("1. Network Map Data (Bus by Bus)")
                df_audit_safe = plot_interactive_network(
                    decisions, 
                    dash_data['network_map'], 
                    loads, 
                    prof, 
                    unique_key=f"map_audit_tab_{time.time()}",
                    plot_component=False  # Apenas retorna o DataFrame, não plota
                )
                if df_audit_safe is not None and not df_audit_safe.empty:
                    st.dataframe(df_audit_safe, width="stretch")
                    
                    # Verificação Rápida para o usuário
                    neg_bal = df_audit_safe[df_audit_safe['Net Balance (kWh)'] < 0].shape[0]
                    pos_bal = df_audit_safe[df_audit_safe['Net Balance (kWh)'] > 0].shape[0]
                    st.caption(f"Resume: {pos_bal} buses with positive balance, {neg_bal} buses with negative balance.")
                else:
                    st.warning("No audit data available for the network map.")

                st.markdown("---")

                # 2. Auditoria do Despacho (Gráfico 1)
                st.subheader("2. Dispatch Chart Data (System Wide)")
                # Tenta recuperar as variáveis usadas no gráfico de despacho
                # (Elas devem estar acessíveis se foram definidas no mesmo escopo da função app)
                if 'df_plot' in locals():
                    st.dataframe(df_plot, width="stretch")
                elif 'df_disp' in locals(): # Fallback
                    st.dataframe(df_disp, width="stretch")
                else:
                    st.error("Dados de despacho não encontrados na memória.")

                st.markdown("---")

                # 3. Auditoria de Tensão (Gráfico 2)
                st.subheader("3. Voltage Profile Data (Min/Max)")
                if 'df_v_clean' in locals():
                    st.dataframe(df_v_clean.describe(), width="stretch")
                    with st.expander("Full Voltage Data Table"):
                        st.dataframe(df_v_clean)
                else:
                    st.warning("Dados de tensão não disponíveis.")

                st.markdown("---")

                # 4. Auditoria Financeira (Gráfico 3 e 4)
                st.subheader("4. Financial Data (Revenue Composition)")
                if 'df_plot_fin' in locals():
                    st.dataframe(df_plot_fin, width="stretch")
                    st.caption("This table shows the resampled financial data used in the revenue charts.")
                elif 'df_fin' in locals():
                    st.dataframe(df_fin, width="stretch")
                
                st.markdown("---")

                # 5. Auditoria Zonal (Gráfico 5)
                st.subheader("5. Zonal Capacity Data")
                if 'df_z' in locals():
                    st.dataframe(df_z, width="stretch")
        else:
            st.info("👈 Adjust parameters and click RUN to start.")
        
    # ==============================================================================
    # SIDEBAR: CONFIGURATION
    # ==============================================================================
    with st.sidebar:
        st.header("⚙️ Configuration Panel")
        # Adicione isso logo abaixo do st.header("⚙️ Configuration Panel")
        st.info(f"🕒 Detected Peak Hours: {identified_peak_hours}")

        tab1, tab2, tab3, tab4 = st.tabs(["Physics", "Econ", "Weights", "Limits"])

        with tab1:
            st.subheader("Simulation Parameters")
            cfg_t = st.number_input("Horizon (hours)", 1, 8760, 24)
            cfg_s_base = st.number_input("S Base (MVA)", 1.0, 100.0, CFG["S_BASE"])
            
            # CORREÇÃO: Penetração em % (0.0 a 1.0)
            cfg_max_pen = st.slider("Max Penetration (%)", 0.0, 1.0, 
                                    float(min(CFG["MAX_PENETRATION"], 0.5)), 0.05)
            
            st.markdown("---")
            st.subheader("🎲 Stochastic Settings")

            # Controles
            in_scenarios = st.slider("Number of Scenarios", 1, 50, 2, help="Caminhos aleatórios simulados.")
            in_sigma_load = st.slider("Load Uncertainty (σ)", 0.0, 0.3, 0.01, format="%.2f")
            in_sigma_ren = st.slider("Renewable Uncertainty (σ)", 0.0, 0.5, 0.01, format="%.2f")
            in_mipgap = st.number_input("Solver Precision (MIPGap)", 0.0000000001, 1.00, 0.0000000001, format="%.10f")
            
            st.markdown("---")
            st.subheader("Voltage Limits (p.u.)")
            c1, c2 = st.columns(2)
            cfg_v_min = c1.number_input("V Min", 0.90, 1.00, CFG["V_MIN"], 0.01)
            cfg_v_max = c2.number_input("V Max", 1.00, 1.10, CFG["V_MAX"], 0.01)

            st.subheader("Optimizer Safety Margins")
            c3, c4 = st.columns(2)
            cfg_opt_v_min = c3.number_input("Opt V Min", 0.90, 1.00, CFG["OPT_V_MIN"], 0.01)
            cfg_opt_v_max = c4.number_input("Opt V Max", 1.00, 1.10, CFG["OPT_V_MAX"], 0.01)

    # --- TAB 2: ECON (DETAILED COSTS & LIFESPANS) ---
        with tab2:
            st.subheader("Financial Parameters (LCOE/LCOS)")
            
            # 1. Taxa de Juros (Discount Rate)
            # O padrão do professor parece ser algo em torno de 5% a 10% nos slides de sensibilidade
            interest_rate = st.number_input("Discount Rate / WACC (%)", 0.0, 50.0, 2.0, 0.5, 
                                        help="Discount rate used in the CRF calculation (Capital Recovery Factor).") / 100.0
            
            st.subheader("Detailed Cost Parameters")
            st.caption("CAPEX ($/unit), O&M ($/unit-yr) and Lifespan (Years)")

            # --- SOLAR PV ---
            with st.expander("☀️ Solar PV Parameters", expanded=True):
                c_pv1, c_pv2 = st.columns(2)
                # CORREÇÃO: Valor de mercado ~1000 $/kW (não 10)
                capex_pv = c_pv1.number_input("PV CAPEX ($/kW)", 0, 5000, 700)
                # Vida útil padrão de painéis é 25 anos
                life_pv  = c_pv2.number_input("PV Lifespan (Years)", 1, 40, 25)
                # O&M costuma ser $10 a $15 por ano
                om_pv = st.number_input("PV O&M ($/kW-yr)", 0, 500, 10)
            
            # --- WIND ---
            with st.expander("🌬️ Wind Parameters", expanded=False):
                c_wd1, c_wd2 = st.columns(2)
                # CORREÇÃO: Valor de mercado ~1500 $/kW (não 15)
                capex_wind = c_wd1.number_input("Wind CAPEX ($/kW)", 0, 5000, 1000)
                life_wind  = c_wd2.number_input("Wind Lifespan (Years)", 1, 40, 20)
                om_wind = st.number_input("Wind O&M ($/kW-yr)", 0, 500, 20)
                
            # --- BATTERY ---
            with st.expander("🔋 Battery (BESS)", expanded=False):
                st.markdown("**Fixed Costs (Hardware)**")
                c_bat1, c_bat2, c_bat3 = st.columns(3)
                # Conversor/Inversor (~$300/kW)
                capex_bess_p = c_bat1.number_input("Power CAPEX ($/kW)", 0, 5000, 200)
                # Células (~$400/kWh para sistemas industriais instalados)
                capex_bess_e = c_bat2.number_input("Energy CAPEX ($/kWh)", 0, 2000, 250)
                # Vida útil de prateleira (Shelf Life) ~15 anos
                life_bess    = c_bat3.number_input("Shelf Life (Years)", 1, 25, 15)
                
                st.markdown("**Variable Costs (Cycling)**")
                c_deg1, c_deg2 = st.columns(2)
                cycles = c_deg1.number_input("Cycle Life", 500, 20000, 6000)
                dod = c_deg2.number_input("DoD (%)", 10, 100, 80) / 100.0
                
                om_bess = st.number_input("BESS O&M ($/kW-yr)", 0, 500, 5)
                
                # Custo de Degradação ($/kWh throughput)
                # Isso é adicionado ao custo de operação para evitar uso desnecessário
                deg_cost_kwh = capex_bess_e / (cycles * dod * 0.95)

            # Configurações de Penalidade e Objetivo
            st.markdown("---")
            st.subheader("⚠️ Constraint Management")
            ignore_penalties = st.checkbox("Ignore Grid Penalties (Pure Financial Sim)", value=False)
            
            if ignore_penalties:
                cfg_penalty = 0.0
                st.warning("Penalidades desligadas! Otimização puramente econômica.")
            else:
                val_atual = float(CFG["PENALTY"])
                if val_atual < 1e6: 
                    val_atual = 1e9
                
                cfg_penalty = st.number_input("Constraint Penalty ($)", 1e6, 1e12, val_atual, format="%.1e")

            st.markdown("---")
            st.subheader("🎯 Optimization Objective")
            opt_mode_sel = st.radio("Target:", ["Investor Profit (Max Revenue)", "Grid Efficiency (Min System Cost)"], index=0)
            opt_mode_code = "PROFIT" if "Investor" in opt_mode_sel else "GRID_COST"

            cfg_alpha = st.slider("Alpha (% of LMP)", 0.0, 1.0, CFG["REMUNERATION"]["ALPHA"], 0.05)

        with tab3:
            st.subheader("Objective Weights (The 'Pie')")
            st.caption("These define the MAX incentive available relative to Energy Price.")
            
            # Pesos Globais (0.0 a 1.0)
            w_soc_val = st.slider("Social Weight (W_SOCIAL)", 0.0, 1.0, 0.25, 0.05, help="Total incentive pot for Social Equity.")
            w_env_val = st.slider("Env Weight (W_ENV)", 0.0, 1.0, 0.25, 0.05)
            w_grid_val = st.slider("Grid Weight (W_GRID)", 0.0, 1.0, 0.25, 0.05)
            w_cap_val = st.slider("Capacity Weight (W_CAPACITY)", 0.0, 1.0, 0.25, 0.05)

            st.markdown("---")
            st.subheader("📍 Social Incentive Allocation")
            st.caption("How the Social Weight is distributed among zones (Must sum ~100%)")
            
            c_soc1, c_soc2, c_soc3 = st.columns(3)
            # Inputs de 0 a 100%
            alloc_rur = c_soc1.number_input("Rural Allocation %", 0, 100, 100, help="Percentage of W_SOCIAL given to Rural nodes.")
            alloc_mix = c_soc2.number_input("Mixed Allocation %", 0, 100, 0, help="Percentage of W_SOCIAL given to Mixed nodes.")
            alloc_urb = c_soc3.number_input("Urban Allocation %", 0, 100, 0, help="Percentage of W_SOCIAL given to Urban nodes.")
            
            # Normalização Visual (apenas para feedback)
            total_alloc = alloc_rur + alloc_mix + alloc_urb
            if total_alloc == 0: total_alloc = 1 # Evita div/0
            st.progress(min(total_alloc, 100)/100.0)
            if total_alloc != 100:
                st.warning(f"Total Allocation is {total_alloc}%. It will be normalized to 100% in simulation.")

        with tab4:
            st.subheader("1. Capacity Allocation Limits")
            st.caption("Physical Constraints: Min/Max % of Total kW installed per zone.")
            
            # Limites Físicos (Isso ainda é útil para forçar instalação em certas áreas)
            lim_z_urb = st.slider("Urban Capacity %", 0.0, 1.0, (0.0, 1.0))
            lim_z_mix = st.slider("Mixed Capacity %", 0.0, 1.0, (0.0, 1.0))
            lim_z_rur = st.slider("Rural Capacity %", 0.0, 1.0, (0.0, 1.0))

            # --- SEÇÃO REMOVIDA: Social Benefit Distribution ---
            # A distribuição agora é controlada pelos inputs "Allocation %" na Tab 3.
            
            st.markdown("---")
            st.subheader("2. Revenue Share Constraints")
            st.caption("Global Constraints: Min/Max % of Total Revenue coming from each source.")
            
            # Limites Financeiros Globais
            lim_rev_soc = st.slider("Social Revenue Share", 0.0, 1.0, (0.0, 1.0))
            lim_rev_env = st.slider("Env Revenue Share", 0.0, 1.0, (0.0, 1.0))
            lim_rev_grid = st.slider("Grid Revenue Share", 0.0, 1.0, (0.0, 1.0))
            lim_rev_cap = st.slider("Capacity Revenue Share", 0.0, 1.0, (0.0, 1.0))

        st.markdown("---")
        run_btn = st.button("RUN SIMULATION", type="primary", width="stretch")

        st.markdown("---")
        st.header("🧪 Batch Sensitivity Analysis")     
        # --- NOVO CONTROLE DE PASSOS (POWERS OF 2) ---
        step_strategy = st.select_slider(
            "Precision Strategy",
            options=["1. Exploratory (0.5)", "2. Refined (0.25)", "3. Detailed (0.125)", "4. Ultra (0.0625)"],
            value="1. Exploratory (0.5)",
            help="Start with 0.5. Then select 0.25 to fill gaps without re-running existing scenarios."
        )
        
        # Define os valores matemáticos exatos
        if "0.5" in step_strategy:   step_val = 0.5
        elif "0.25" in step_strategy: step_val = 0.25
        elif "0.125" in step_strategy: step_val = 0.125
        else: step_val = 0.0625
        
        # Gera lista de valores (0.0, 0.5, 1.0...)
        raw_vals = np.arange(0.0, 1.0001, step_val)
        search_values = [round(x, 4) for x in raw_vals]
        
        st.caption(f"Values: {search_values}")
        st.caption(f"Total Combs: {len(search_values)**4}")
        
        run_batch_btn = st.button("🚀 RUN BATCH (Smart Resume)", type="secondary", width="stretch")

    # ==============================================================================
    # EXECUTION LOGIC
    # ==============================================================================

    if 'sim_results' not in st.session_state:
        st.session_state.sim_results = None
    if 'loaded_from_history' not in st.session_state:
        st.session_state.loaded_from_history = False

    HISTORY_DIR = os.path.join(CFG["OUTPUT_DIR"], "history_data")
    if not os.path.exists(HISTORY_DIR): os.makedirs(HISTORY_DIR)
    history_csv = os.path.join(CFG["OUTPUT_DIR"], "Optimization_History_Log.csv")

    # ==============================================================================
    # CONTROLE DE EXECUÇÃO (MAIN)
    # ==============================================================================

    # --- 1. MODO MANUAL (Um clique, um resultado) ---
    if run_btn:
        # Chama a função direto na raiz (desenha e fica na tela)
        rodar_cenario_completo(w_grid_val, w_cap_val, w_soc_val, w_env_val, identified_peak_hours)


    # ==============================================================================
    # 2. MODO FÁBRICA (BATCH) - COM VERIFICAÇÃO NO CSV
    # ==============================================================================
    if run_batch_btn:
        st.markdown("---")
        st.header("🚀 Batch Simulation")
        
        eixo = search_values
        
        # Gera todas as combinações matemáticas possíveis (Grid, Cap, Soc, Env)
        combinacoes = list(product(eixo, repeat=4)) 
        total_teorico = len(combinacoes)
        
        # ------------------------------------------------------------------
        # PASSO A: Ler o arquivo CSV para ver o que já foi feito
        # ------------------------------------------------------------------
        arquivo_master = os.path.join(CFG["OUTPUT_DIR"], "MASTER_Hourly_Results.csv")
        combinacoes_existentes = set()
        
        st.write(f"📂 Checking existing file: `{arquivo_master}`")
        
        if os.path.exists(arquivo_master):
            try:
                # Carrega APENAS as colunas de peso para ser muito rápido (usa pouca memória)
                # Nomes exatos que você pediu: Weight_Cap, Weight_Env, Weight_Grid, Weight_Social
                cols_to_check = ["Weight_Grid", "Weight_Cap", "Weight_Social", "Weight_Env"]
                
                # Lê o CSV
                df_check = pd.read_csv(arquivo_master, usecols=cols_to_check)
                
                # Arredonda para 2 casas decimais para evitar erros de precisão (0.33333 != 0.33)
                df_check = df_check.round(2)
                
                # Transforma em um conjunto de tuplas únicas (Remove duplicatas automaticamente)
                # A ordem aqui tem que ser a mesma do loop: Grid, Cap, Soc, Env
                for _, row in df_check.iterrows():
                    tupla = (
                        row["Weight_Grid"], 
                        row["Weight_Cap"], 
                        row["Weight_Social"], 
                        row["Weight_Env"]
                    )
                    combinacoes_existentes.add(tupla)
                    
                st.info(f"✅ Master file found. {len(combinacoes_existentes)} existing combinations loaded.")
                
            except Exception as e:
                st.warning(f"⚠️ Arquivo existe mas não consegui ler as colunas de peso. Vou rodar tudo. Erro: {e}")
        else:
            st.info("ℹ️ Arquivo Master ainda não existe. Será criado agora.")

        # ------------------------------------------------------------------
        # PASSO B: Executar o Loop filtrando pelo que achamos no CSV
        # ------------------------------------------------------------------
        barra_progresso = st.progress(0.0)
        texto_status = st.empty()
        area_de_desenho = st.empty()
        
        contador = 0
        pulados = 0
        calculados = 0
        inicio = time.time()
        
        for g, c, s, e in combinacoes:
            contador += 1
            barra_progresso.progress(contador / total_teorico)
            
            # Cria a assinatura atual (Arredondada para bater com o CSV)
            assinatura_atual = (round(g, 2), round(c, 2), round(s, 2), round(e, 2))
            
            msg = f"Scenario {contador}/{total_teorico} | G:{g:.2f} C:{c:.2f} S:{s:.2f} E:{e:.2f}"
            
            # >>> AQUI É O PULO DO GATO (CHECKAGEM) <<<
            if assinatura_atual in combinacoes_existentes:
                # Se já está no set, apenas avisamos e pulamos
                pulados += 1
                texto_status.warning(f"{msg} -> ⏭️ Already exists in CSV. Skipping...")
                # Não faz sleep aqui para ser instantâneo
                continue 
            
            # Se não existe, rodamos o DEF
            calculados += 1
            peak = identified_peak_hours
            
            with area_de_desenho.container():
                
                status_texto = st.empty()

                # Chama o seu def original sem mudar nada nele
                for g, c, s, e in combinacoes:
                    status_texto.info(f"Scenario {contador+1}/{total_teorico} | G:{g:.2f} C:{c:.2f} S:{s:.2f} E:{e:.2f} → ⚙️ Running...")
                    try:
                        rodar_cenario_completo(g, c, s, e, peak)
                        calculados += 1
                        status_texto.success(f"✅ Scenario {contador+1} OK | Total success: {calculados}")

                    except KeyError as ke:
                        pulados += 1
                        status_texto.warning(f"⚠️ KeyError '{ke}' (G:{g:.2f} C:{c:.2f} S:{s:.2f} E:{e:.2f}) | Skipped: {pulados}")
                        print(f"KeyError '{ke}' in scenario (G:{g:.2f} C:{c:.2f} S:{s:.2f} E:{e:.2f}). Skipping...")
                        # Opcional: log em arquivo ou CSV de erros
                        continue
                    except Exception as ex:
                        pulados += 1
                        status_texto.error(f"❌ Error: {ex} (G:{g:.2f} C:{c:.2f} S:{s:.2f} E:{e:.2f}) | Skipped: {pulados}")
                        print(f"Unexpected error in scenario (G:{g:.2f} C:{c:.2f} S:{s:.2f} E:{e:.2f}): {ex}. Skipping...")
                        continue
                

                    contador += 1
                    barra_progresso.progress(contador / total_teorico)

            # Final status
            status_texto.success(f"🎉 Batch completed! Success: {calculados} | Skipped: {pulados} | Total: {contador}")
            print(f"Batch finished: {calculados} success, {pulados} skipped")  # Final terminal log


            # Adiciona na lista de existentes para não repetir se aparecer de novo (redundância)
            combinacoes_existentes.add(assinatura_atual)
            
            # Pausa para salvar arquivo com segurança
            time.sleep(0.5)

        tempo_total = time.time() - inicio
        area_de_desenho.empty()
        
        st.success(f"""
        🏁 END! Process Completed.
        - Total checked: {total_teorico}
        - ⏭️ Skipped (Already in CSV): {pulados}
        - ⚙️ Calculated Now: {calculados}
        - Total Time: {tempo_total:.1f}s
        """)
        st.balloons()

    st.markdown("---")
    st.subheader("📑 Report update")
    
    if st.button("🔄 Update Global Report (Index.py)", width="stretch", help="Run the index.py script to recalculate scores and indicators."):
        
        # 1. Pega o caminho exato onde ESTE arquivo está
        CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
        
        # 2. Aponta para o index.py que está AO LADO dele (infalível)
        script_path = os.path.join(CURRENT_DIR, "index.py")
        
        

        if 'identified_peak_hours' in locals() or 'identified_peak_hours' in globals():
            peaks_str = ",".join(map(str, identified_peak_hours))
        else:
            peaks_str = "15,16,17,18" # Fallback se não encontrar
        
        # Feedback visual enquanto roda
        with st.spinner(f"Executing Post-Processing (Flow, Scores, KPIs) with {peaks_str}..."):
            try:              
                # Executa o arquivo index.py usando o mesmo interpretador Python atual
                result = subprocess.run(
                    [sys.executable, script_path],
                    cwd=CURRENT_DIR,
                    capture_output=True, # Captura o que o script imprimir (print)
                    text=True
                )
                
                if result.returncode == 0:
                    st.success("Script executed successfully.")
                    # Mostra o log do index.py (opcional, bom para debug)
                    with st.expander("View Output Log"):
                        st.code(result.stdout)
                else:
                    st.error("Script execution failed.")
                    with st.expander("View Error Log"):
                        st.code(result.stderr)
                        
            except Exception as e:
                st.error(f"Error executing script: {e}")

if __name__ == "__main__":
    app()
            
            ### streamlit run "./code/dashboard_batch.py"