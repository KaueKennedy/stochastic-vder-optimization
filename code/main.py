print(">>> 1. INICIO DO SCRIPT DE DIAGNOSTICO <<<")
import sys
import os
import time
import traceback
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

# 1. Configurar Caminhos
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(current_dir)
    print(f">>> 2. Diretório do script: {current_dir}")
except Exception as e:
    print(f"ERRO CRÍTICO AO CONFIGURAR CAMINHOS: {e}")

# 2. Tentar Importar Módulos (Um por um para achar o culpado)
try:
    print(">>> 3. Tentando importar config.py...")
    from config import CFG
    print(f"    -> Config carregado. Pasta de dados: {os.path.exists(CFG['EXCEL_REDE'])}")
    
    print(">>> 4. Tentando importar data_loader...")
    import data_loader
    
    print(">>> 5. Tentando importar optimizer...")
    import optimizer
    
    print(">>> 6. Tentando importar simulation...")
    import simulation
    
    print(">>> 7. Tentando importar reporting...")
    import reporting
    
except ImportError as ie:
    print(f"\n[ERRO DE IMPORTAÇÃO]: O Python não achou um arquivo.")
    print(f"Detalhe: {ie}")
    print("Verifique se todos os arquivos .py estão na mesma pasta que o main.py")
    input("Pressione Enter para sair...")
    sys.exit(1)
except Exception as e:
    print(f"\n[ERRO DE SINTAXE OU OUTRO]: Um dos arquivos tem erro no código.")
    print(f"Detalhe: {e}")
    traceback.print_exc()
    input("Pressione Enter para sair...")
    sys.exit(1)

def main():
    print("\n>>> 8. Entrando na função main() <<<")
    print("="*50)
    print("   SISTEMA DE OTIMIZAÇÃO E ANÁLISE DE DERs")
    print("="*50)
    
    start_time = time.time()

    try:
        # 1. CARREGAMENTO
        print(">>> Chamando data_loader.load_inputs()...")
        lines, buses, loads, prof = data_loader.load_inputs()
        
        # 2. OTIMIZAÇÃO
        print(">>> Chamando optimizer.run_optimization()...")
        decisions, price, dash_data, df_v_opt = optimizer.run_optimization(lines, buses, loads, prof)
        
        if not decisions:
            print("\n[AVISO] O otimizador não instalou nenhum DER.")
            print("Verifique se o CAPEX está muito alto ou a penetração permitida é 0.")
            return

        # 3. SIMULAÇÃO
        print(">>> Chamando simulation.run_digital_twin()...")
        full_data = simulation.run_digital_twin(decisions)
        
        # 4. RELATÓRIOS
        print(">>> Chamando reporting.generate_reports()...")
        reporting.generate_reports(decisions, full_data, dash_data, df_v_opt, buses)
        
        elapsed = time.time() - start_time
        print(f"\n>>> SUCESSO! Tempo total: {elapsed:.2f} segundos.")
        print(f">>> Resultados em: {CFG['OUTPUT_DIR']}")

    except Exception as e:
        print(f"\n[ERRO FATAL DURANTE EXECUÇÃO]:")
        print(e)
        traceback.print_exc()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Erro no bloco main: {e}")
    
    print("\n>>> FIM DO SCRIPT <<<")