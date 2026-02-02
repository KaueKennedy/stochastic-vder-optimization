import opendssdirect as dss
import pandas as pd
import os
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

def export_grid_to_excel(path_to_dss, output_excel_name="Rede_240Bus_Dados.xlsx"):
    """
    Carrega o OpenDSS e exporta a topologia para um Excel estruturado
    exatamente como o otimizador precisa.
    """
    print(f"--- Iniciando Extração do OpenDSS: {path_to_dss} ---")

    # 1. Carregar o Circuito
    dss.Basic.ClearAll()
    dss.Command(f"Compile [{path_to_dss}]")
    
    if not dss.Circuit.Name():
        raise RuntimeError("❌ Erro: O OpenDSS não conseguiu compilar o circuito. Verifique o caminho do arquivo.")

    print(f"   -> Circuito Carregado: {dss.Circuit.Name()}")

    # ==========================================================================
    # 1. EXTRAÇÃO DE BARRAS (Tab: Buses)
    # ==========================================================================
    print("   -> Extraindo Barras (Buses)...")
    bus_names = dss.Circuit.AllBusNames()
    bus_data = []

    for bus in bus_names:
        dss.Circuit.SetActiveBus(bus)
        # Atenção: OpenDSS retorna kV base fase-neutro ou fase-fase dependendo da definição
        kv_base = dss.Bus.kVBase()
        x = dss.Bus.X()
        y = dss.Bus.Y()
        
        bus_data.append({
            "BusName": bus.lower(), # Força minúsculo para padronizar
            "kVBase": kv_base,
            "x": x,
            "y": y,
            "Zone": "Mixed" # Default
        })
    
    df_buses = pd.DataFrame(bus_data)

    # ==========================================================================
    # 2. EXTRAÇÃO DE LINHAS (Tab: Lines)
    # ==========================================================================
    print("   -> Extraindo Linhas (Lines)...")
    line_data = []
    
    lines_list = dss.Lines.AllNames()
    for line in lines_list:
        dss.Lines.Name(line)
        
        # Limpeza: remove os nós (.1.2.3) para pegar só o nome da barra
        bus1 = dss.Lines.Bus1().split('.')[0].lower()
        bus2 = dss.Lines.Bus2().split('.')[0].lower()
        
        length = dss.Lines.Length()
        
        # Propriedades (R1 e X1 são Sequência Positiva)
        r1 = dss.Lines.R1()
        x1 = dss.Lines.X1()
        
        # OpenDSS geralmente retorna R1 em Ohms/unidade de comprimento
        # Mas em alguns modelos pode já ser total. Vamos multiplicar pelo length por segurança padrão.
        # Se os valores ficarem muito altos no otimizador, significa que o DSS já dava o total.
        r_total = r1 * length
        x_total = x1 * length
        
        norm_amps = dss.Lines.NormAmps()
        
        line_data.append({
            "Name": line,
            "Bus1": bus1,
            "Bus2": bus2,
            "R_Total_Ohms_Calc": r_total,
            "X_Total_Ohms_Calc": x_total,
            "NormAmps": norm_amps
        })
        
    df_lines = pd.DataFrame(line_data)

    # ==========================================================================
    # 3. EXTRAÇÃO DE TRANSFORMADORES (Tab: Transformers)
    # ==========================================================================
    print("   -> Extraindo Transformadores (Transformers)...")
    trafo_data = []
    
    trafos_list = dss.Transformers.AllNames()
    for tr in trafos_list:
        dss.Transformers.Name(tr)
        
        # Barras
        buses = dss.CktElement.BusNames()
        if len(buses) >= 2:
            bus1 = buses[0].split('.')[0].lower()
            bus2 = buses[1].split('.')[0].lower()
        else:
            bus1 = buses[0].split('.')[0].lower()
            bus2 = ""
        
        kva = dss.Transformers.kVA()
        xhl = dss.Transformers.Xhl()

        if xhl < 0.1:
            print(f"      ⚠️ Warning: xhl very low ({xhl}%) for transformer {tr}. Setting to 1.0%")
            xhl = 1.0
        
        trafo_data.append({
            "Name": tr,
            "Bus1_High": bus1,
            "Bus2_Low": bus2,
            "kVA": kva,
            "XHL_Percent": xhl
        })
        
    df_trafos = pd.DataFrame(trafo_data)

    # ==========================================================================
    # 4. EXTRAÇÃO DE CARGAS (Tab: Loads)
    # ==========================================================================
    print("   -> Extraindo Cargas (Loads)...")
    load_data = []
    
    loads_list = dss.Loads.AllNames()
    for ld in loads_list:
        dss.Loads.Name(ld)
        
        bus_raw = dss.CktElement.BusNames()[0]
        bus = bus_raw.split('.')[0].lower()
        kw = dss.Loads.kW()
        kv = dss.Loads.kV()
        
        load_data.append({
            "Name": ld,
            "Bus": bus,
            "kW": kw,
            "kV": kv
        })
        
    df_loads = pd.DataFrame(load_data)

    # ==========================================================================
    # 5. SALVAR ARQUIVO EXCEL
    # ==========================================================================
    print(f"   -> Salvando arquivo Excel: {output_excel_name}")
    
    try:
        with pd.ExcelWriter(output_excel_name, engine='openpyxl') as writer:
            df_lines.to_excel(writer, sheet_name='Lines', index=False)
            df_trafos.to_excel(writer, sheet_name='Transformers', index=False)
            df_buses.to_excel(writer, sheet_name='Buses', index=False)
            df_loads.to_excel(writer, sheet_name='Loads', index=False)
        print("   ✅ Sucesso! Arquivo gerado.")
    except Exception as e:
        print(f"   ❌ Erro ao salvar Excel: {e}")

    return df_lines, df_trafos, df_buses, df_loads

# ==============================================================================
# EXECUÇÃO
# ==============================================================================
if __name__ == "__main__":
    # Caminho exato baseado no seu log de erro
    MEU_DSS = "C:/Users/KKCOD/OneDrive - Université Laval/Recherche/codes/Iowa_Distribution_Test_Systems/OpenDSS Model/OpenDSS Model/master.dss"
    
    if os.path.exists(MEU_DSS):
        export_grid_to_excel(MEU_DSS)
    else:
        print(f"❌ Arquivo não encontrado: {MEU_DSS}")
        print("Verifique o caminho e as barras (/).")