import py_dss_interface
import ctypes
import os
import math
import pandas as pd
import numpy as np
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

def get_safe_path(path):
    if not os.path.exists(path): return path
    buf = ctypes.create_unicode_buffer(260)
    _ = ctypes.windll.kernel32.GetShortPathNameW(str(path), buf, 260)
    return buf.value

def run_digital_twin(decisions):
    print(f"--- [3] Validando no OpenDSS (Extração Massiva: V, I, P, Q) ---")
    
    dss = py_dss_interface.DSS()
    dss_file_safe = get_safe_path(CFG['DSS_FILE'])
    dss.text(f"compile [{dss_file_safe}]")
    
    # 1. Instalação dos DERs
    print(f"   -> Instalando {len(decisions)} DERs...")
    for bus, data in decisions.items():
        kw = data['cap_kw']
        curve = data['profile']
        
        if kw > 0:
            mult_array = [v/kw for v in curve]
        else:
            mult_array = [0] * CFG['T']
            
        mult_str = str(mult_array).replace('[','').replace(']','')
        shape_name = f"shape_{bus}"
        
        dss.text(f"New LoadShape.{shape_name} npts={CFG['T']} interval=1 mult=[{mult_str}]")
        
        dss.circuit.set_active_bus(bus)
        kv = dss.bus.kv_base
        phases = 3 if dss.bus.num_nodes >= 3 else 1
        bus_conn = f"{bus}.1.2.3" if phases==3 else f"{bus}.1"
        kv_gen = kv if phases==3 else kv/1.732
        
        dss.text(f"New Generator.DER_{bus} phases={phases} bus1={bus_conn} kV={kv_gen} kW={kw} daily={shape_name} model=1")

    # 2. Simulação Temporal
    print("   -> Rodando Fluxo de Potência e Extraindo Dados...")
    dss.text(f"Set Mode=Daily number={CFG['T']} stepsize=1h")
    
    # --- ESTRUTURAS DE ARMAZENAMENTO ---
    history_voltages = []
    history_global = []
    history_flows_P = []
    history_flows_Q = []
    history_currents = [] 
    
    all_buses = dss.circuit.buses_names
    
    for t in range(CFG['T']):
        dss.text("Set number=1")
        dss.text("Solve")
        
        # A. TENSÕES (Todas as Barras)
        row_v = {'Hora': t}
        v_values_only = [] 
        
        for bus in all_buses:
            dss.circuit.set_active_bus(bus)
            vc = dss.bus.pu_voltages
            val = 0.0
            if vc:
                mags = []
                for i in range(0, len(vc), 2):
                    m = math.sqrt(vc[i]**2 + vc[i+1]**2)
                    if m > 0.1: mags.append(m)
                val = max(mags) if mags else 0.0
            
            row_v[bus] = val
            if val > 0.1: v_values_only.append(val)
            
        history_voltages.append(row_v)

        # B. FLUXOS E CORRENTES (Todas as Linhas)
        row_fp = {'Hora': t}
        row_fq = {'Hora': t}
        row_fi = {'Hora': t}
        
        dss.lines.first()
        for _ in range(dss.lines.count):
            lname = dss.lines.name
            dss.circuit.set_active_element(f"Line.{lname}")
            
            # Potência
            powers = dss.cktelement.powers
            n_ph = dss.cktelement.num_phases
            p_in = sum(powers[i] for i in range(0, 2*n_ph, 2))
            q_in = sum(powers[i] for i in range(1, 2*n_ph, 2))
            
            # Corrente
            cur_mag_ang = dss.cktelement.currents_mag_ang
            if cur_mag_ang:
                mags = [cur_mag_ang[i] for i in range(0, 2*n_ph, 2)]
                i_max = max(mags) if mags else 0
            else:
                i_max = 0
            
            row_fp[lname] = p_in
            row_fq[lname] = q_in
            row_fi[lname] = i_max
            
            dss.lines.next()
            
        history_flows_P.append(row_fp)
        history_flows_Q.append(row_fq)
        history_currents.append(row_fi)

        # C. DADOS GLOBAIS (CORRIGIDO AQUI)
        losses = dss.circuit.losses # [Watts, VArs]
        p_loss = losses[0] / 1000   # kW
        q_loss = losses[1] / 1000   # kvar
        
        row_g = {
            'Hora': t,
            'Tensao_Max': max(v_values_only) if v_values_only else 0,
            'Tensao_Min': min(v_values_only) if v_values_only else 0,
            'Perdas_Ativas_kW': p_loss,      # <--- NOME CORRIGIDO
            'Perdas_Reativas_kvar': q_loss   # <--- ADICIONADO
        }
        history_global.append(row_g)

    # --- RETORNA DICIONÁRIO COMPLETO ---
    full_data = {
        'voltages': pd.DataFrame(history_voltages),
        'global': pd.DataFrame(history_global),
        'flows_P': pd.DataFrame(history_flows_P),
        'flows_Q': pd.DataFrame(history_flows_Q),
        'currents': pd.DataFrame(history_currents)
    }
    
    return full_data