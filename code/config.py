import os

# ==============================================================================
# 1. NAVEGAÇÃO DE DIRETÓRIOS
# ==============================================================================
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CODE_DIR)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "resultados")
DATA_DIR = os.path.join(PROJECT_ROOT, "Iowa_Distribution_Test_Systems", "OpenDSS Model", "OpenDSS Model")

# ==============================================================================
# 2. DICIONÁRIO DE CONFIGURAÇÃO (CFG)
# ==============================================================================
CFG = {
    # --- Parâmetros de Simulação ---
    "T": 24,                   
    "S_BASE": 10.0,            
    
    # LIMITES FÍSICOS
    "V_MIN": 0.95,             
    "V_MAX": 1.05,             

    # MARGENS DE SEGURANÇA DO OTIMIZADOR
    "OPT_V_MIN": 0.96,
    "OPT_V_MAX": 1.04,

    # --- PARÂMETROS DE SIMULAÇÃO ESTOCÁSTICA ---
    "STOCHASTIC": {
            "NUM_SCENARIOS": 10,
            "SIGMA_LOAD": 0.05,
            "SIGMA_RENEWABLE": 0.10,
            "MIP_GAP": 0.001
        },
    # --- PARÂMETROS ECONÔMICOS DETALHADOS (Default) ---
    # Valores calculados com base nos defaults do Dashboard:
    # PV: 25 anos | Wind: 20 anos | Bateria: 15 anos (Shelf), 6000 ciclos, 80% DoD
    "COSTS": {
        "PV": {
            "CAPEX": 1000, 
            "OM": 15, 
            "AMORT": 1.0/(25*365)  # ~0.0001096
        },
        "WIND": {
            "CAPEX": 1500, 
            "OM": 25, 
            "AMORT": 1.0/(20*365)  # ~0.0001370
        },
        "BESS": {
            "CAPEX_P": 300,        # Inversor $/kW
            "CAPEX_E": 400,        # Energia $/kWh
            "OM": 10,
            "AMORT": 1.0/(15*365), # Amortização Fixa (Tempo)
            "DEG_COST": 0.0877     # Custo Variável $/kWh (400 / (6000*0.8*0.95))
        }
    },
    
    # CAPEX Genérico (Zero para forçar uso do detalhado, mantido para compatibilidade)
    "CAPEX_KW": 0, 
    "CAPEX_FACTOR": 1/(15*365),
    
    "PENALTY": 1e9,            
    "MAX_PENETRATION": 0.50, # 50%
    "SLACK_BUS_NAME": "eq_source_bus",

    # --- DEFINIÇÃO DE VULNERABILIDADE SOCIAL (SCORING) ---
    "SOCIAL_SCORING": {
        "Rural": {"base": 7.0, "sigma": 1.5},
        "Mixed": {"base": 5.0, "sigma": 1.5},
        "Urban": {"base": 3.0, "sigma": 1.5},
        "Default": {"base": 2.0, "sigma": 1.0}
    },

    # --- REMUNERAÇÃO (PESOS E RESTRIÇÕES) ---
    "REMUNERATION": {
        "ALPHA": 0.30, # 30% do LMP
        
        # Estrutura: "val" (Peso Objetivo) | "min/max" (Restrição de % da Receita)
        "W_SOCIAL":   {"val": 0.5, "min": 0.0, "max": 1.0},
        "W_ENV":      {"val": 0.3, "min": 0.0, "max": 1.0},
        "W_GRID":     {"val": 0.2, "min": 0.0, "max": 1.0},
        "W_CAPACITY": {"val": 1.0, "min": 0.0, "max": 1.0},
        
        "PEAK_HOURS": [], # Preenchido dinamicamente
        "CHART_GROUPING": "1H"
    },

    # --- LIMITES DE ALOCAÇÃO FÍSICA (Zonal Capacity) ---
    "ZONAL_LIMITS": {
        "Urban": {"min": 0.0, "max": 1.0},
        "Mixed": {"min": 0.0, "max": 1.0},
        "Rural": {"min": 0.0, "max": 1.0}
    },

    # --- LIMITES DE DISTRIBUIÇÃO FINANCEIRA (Social Benefit Share) ---
    "SOCIAL_DISTRIBUTION_LIMITS": {
        "Urban": {"min": 0.0, "max": 1.0},
        "Mixed": {"min": 0.0, "max": 1.0},
        "Rural": {"min": 0.0, "max": 1.0}
    },

    # --- Caminhos ---
    "OUTPUT_DIR": OUTPUT_DIR,
    "EXCEL_REDE": os.path.join(DATA_DIR, "Rede_240Bus_Dados.xlsx"),
    "DSS_FILE": os.path.join(DATA_DIR, "Master.dss"),
    
    # --- Perfis ---
    "FILE_LOAD": os.path.join(DATA_DIR, "perfil_carga_real.csv"),
    "FILE_WIND": os.path.join(DATA_DIR, "perfil_eolico_fixo.csv"),
    "FILE_PRICE": os.path.join(DATA_DIR, "perfil_precos_real.csv"),
    "FILE_SOLAR": os.path.join(DATA_DIR, "perfil_solar_fixo.csv"),
}


_t_sim = CFG["T"]

if _t_sim <= 24:
    CFG["REMUNERATION"]["CHART_GROUPING"] = "1H"
elif _t_sim <= 729:
    CFG["REMUNERATION"]["CHART_GROUPING"] = "1D"  # Equivalente a 24h
else:
    CFG["REMUNERATION"]["CHART_GROUPING"] = "30D" # Equivalente a 730h

print(f"[CONFIG] Horizonte T={_t_sim}. Agrupamento de Gráfico definido para: {CFG['REMUNERATION']['CHART_GROUPING']}")