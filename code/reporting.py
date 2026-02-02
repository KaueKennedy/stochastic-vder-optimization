import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans
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

def classify_zones_kmeans(df_buses):
    print("   -> Executando Clustering K-Means com Correção de Coordenadas...")
    
    # 1. Criar um Dicionário de Coordenadas Válidas (Busca rápida)
    # Filtra apenas barras que TEM coordenadas (diferentes de zero)
    valid_map = df_buses[df_buses['x'] != 0].set_index('BusName')[['x', 'y']].to_dict('index')
    
    # Listas para armazenar os dados corrigidos
    corrected_coords = []
    bus_names = []
    
    # 2. Corrigir Coordenadas (Herança de Pai)
    for _, row in df_buses.iterrows():
        bus = str(row['BusName']).strip()
        x, y = row['x'], row['y']
        
        # Se a coordenada for (0,0) e não for a fonte (eq_source), tenta achar o pai
        if abs(x) < 0.1 and abs(y) < 0.1 and 'source' not in bus:
            # Tenta limpar o nome para achar o pai
            # Ex: "t_bus1003_l" -> vira "bus1003"
            clean_name = bus.replace('t_', '').replace('_l', '').strip()
            
            if clean_name in valid_map:
                x = valid_map[clean_name]['x']
                y = valid_map[clean_name]['y']
                # print(f"      [FIX] Barra {bus} herdou coords de {clean_name}: ({x}, {y})")
        
        corrected_coords.append([x, y])
        bus_names.append(bus)

    # Converte para matriz numpy para o K-Means
    X = np.array(corrected_coords)
    
    # 3. Aplicar K-Means
    # Se houver menos de 3 pontos distintos, reduz clusters para evitar erro
    n_unique = len(np.unique(X, axis=0))
    k = min(3, n_unique)
    
    if k < 3:
        print(f"   [AVISO] Poucas coordenadas distintas ({n_unique}). Usando k={k}.")
        
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    centers = kmeans.cluster_centers_
    
    # 4. Identificar Zonas pela Distância da Origem
    dists_origin = [np.linalg.norm(c) for c in centers]
    sorted_indices = np.argsort(dists_origin) # [Indice Menor Dist, ..., Indice Maior Dist]
    
    # Mapeamento dinâmico (se k=3: Urban, Mixed, Rural. Se k=2: Urban, Rural)
    zone_labels = ['Urban', 'Mixed', 'Rural']
    if k < 3: zone_labels = ['Urban', 'Rural'][:k]
        
    # Cria mapa {Cluster_ID -> Nome da Zona}
    cluster_map = {}
    for i, cluster_idx in enumerate(sorted_indices):
        # Garante que não estoure o índice se k < 3
        label = zone_labels[i] if i < len(zone_labels) else 'Rural'
        cluster_map[cluster_idx] = label
            
    # 5. Retornar Dicionário {BusName: Zone}
    bus_zone_dict = {}
    for i, bus in enumerate(bus_names):
        cluster_id = labels[i]
        bus_zone_dict[bus] = cluster_map[cluster_id]
        
    return bus_zone_dict

def calculate_financial_composition(decisions, df_buses, price_profile, dash_data=None):
    """
    Recupera os dados financeiros calculados pelo otimizador e formata para DataFrame.
    Não utiliza mais o Config para recálculo, garantindo consistência com o solver.
    """
    print("   -> Processando Dados Financeiros (Vindos do Otimizador)...")
    
    # Se não houver dados financeiros passados (compatibilidade)
    if dash_data is None or 'financial_ts' not in dash_data:
        print("   [AVISO] Dados financeiros não encontrados no dash_data.")
        return pd.DataFrame()

    # Recupera o dicionário pronto
    fin_data = dash_data['financial_ts']
    
    # Cria DataFrame Pandas
    df_fin = pd.DataFrame(fin_data)
    
    # Cria índice de tempo
    T = len(df_fin)
    start_date = pd.Timestamp("2024-01-01 00:00")
    
    # Ajuste de frequência baseado no T
    freq = 'h'
    if T > 25 and T <= 730: freq = 'D' # Caso venha diário
    
    df_fin.index = pd.date_range(start=start_date, periods=T, freq=freq)
    
    return df_fin

def plot_financial_composition(df_fin):
    print("   -> Gerando Gráfico de Composição de Renda...")
    
    # 1. Agrupamento conforme Config
    freq = CFG["REMUNERATION"]["CHART_GROUPING"] # ex: '1D', '30D'
    
    # Soma os valores dentro do período (Resample)
    df_grouped = df_fin.resample(freq).sum()
    
    # Se o agrupamento resultou em zero linhas (ex: pedir 30D numa simulacao de 24h),
    # revertemos para o original sem crashar
    if len(df_grouped) == 0:
        print(f"   [AVISO] Agrupamento '{freq}' resultou em dataframe vazio. Usando dados horários.")
        df_grouped = df_fin
        freq_label = "Hourly"
    else:
        freq_label = freq

    # 2. Formatar Eixo X para ficar bonito
    if 'D' in freq:
        x_labels = df_grouped.index.strftime('%Y-%m-%d')
    elif 'H' in freq:
        x_labels = df_grouped.index.strftime('%H:00')
    else:
        x_labels = df_grouped.index.astype(str)

    # 3. Plotagem
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Cores personalizadas para cada componente
    colors = ['#2c3e50', '#27ae60', '#f39c12', '#8e44ad', '#c0392b']
    # Energy, Env, Grid, Social, Capacity
    
    df_grouped.plot(kind='bar', stacked=True, ax=ax, color=colors, width=0.8)
    
    # Estilização
    ax.set_title(f"Composition of DER Revenue (Grouped by {freq_label})\nIncludes Social Vulnerability & Grid Resilience Bonuses", fontsize=14)
    ax.set_ylabel("Revenue ($)", fontsize=12)
    ax.set_xlabel("Time Period", fontsize=12)
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.legend(title="Revenue Source", loc='upper left', bbox_to_anchor=(1, 1))
    
    # Adicionar totais no topo das barras
    totals = df_grouped.sum(axis=1)
    for i, val in enumerate(totals):
        ax.text(i, val, f"${val:.2f}", ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(CFG["OUTPUT_DIR"], "Grafico_Receita_Composta.png"))
    plt.close()
    
    # Salvar CSV
    csv_path = os.path.join(CFG["OUTPUT_DIR"], "Relatorio_Financeiro_Detalhado.csv")
    df_grouped.to_csv(csv_path)
    print(f"   -> Relatório financeiro salvo em: {csv_path}")

def generate_reports(decisions, full_data, dash_data, df_v_opt, df_buses):
    print(f"--- [4] Gerando Relatórios Detalhados ---")
    zone_map = classify_zones_kmeans(df_buses)
    ts = range(CFG['T'])
    tsd = dash_data['ts_data']
    df_v_real = full_data['voltages']
    
    # --- CÁLCULOS PARA O GRÁFICO ---
    # Recupera vetores
    vec_pv = np.array(tsd['PV'])
    vec_wind = np.array(tsd['Wind'])
    vec_dis = np.array(tsd['ESS_Dis'])
    vec_ch = np.array(tsd['ESS_Ch'])
    vec_load = np.array(tsd['Load'])
    vec_soc_kwh = np.array(tsd['SoC'])

    # --- CÁLCULO SOC % ---
    # Capacidade Total em kWh = Capacidade em kW * 4 horas (definido no otimizador)
    total_batt_cap_kwh = dash_data['cap_total'] * 4.0
    
    if total_batt_cap_kwh > 0:
        vec_soc_pct = (vec_soc_kwh / total_batt_cap_kwh) * 100
    else:
        vec_soc_pct = np.zeros(len(ts))
    
    # Calcula o que veio da Rede (Importação)
    # O que precisamos (Carga + Carregar Bateria) - O que geramos (PV + Wind + Descarregar)
    demanda_total = vec_load + vec_ch
    geracao_local = vec_pv + vec_wind + vec_dis
    vec_grid = np.maximum(0, demanda_total - geracao_local)
    
    # Injeção Líquida (para o Excel)
    injecao_liquida = geracao_local - vec_ch - vec_load # Se positivo, exportou. Se negativo, importou.

    # --- CÁLCULO DA PENETRAÇÃO (DEFINIÇÃO DO GRÁFICO) ---
    # Penetração = (Energia DER) / (Energia DER + Energia Importada)
    total_energy_der = np.sum(vec_pv) + np.sum(vec_wind) + np.sum(vec_dis)
    total_energy_grid = np.sum(vec_grid)
    total_energy_supply = total_energy_der + total_energy_grid
    
    if total_energy_supply > 0:
        penetration_val = (total_energy_der / total_energy_supply) * 100
    else:
        penetration_val = 0.0


    # --- 1. EXCEL (Mantido e Atualizado) ---
    excel_path = os.path.join(CFG["OUTPUT_DIR"], "Relatorio_Completo_Sistema.xlsx")
    print(f"   -> Compilando Excel...")
    
    df_balanco = pd.DataFrame({
        'Hora': ts,
        'Preco': tsd['Price'],
        'Carga_Total_kW': vec_load,
        'Rede_Compra_kW': vec_grid,
        'PV_kW': vec_pv,
        'Wind_kW': vec_wind,
        'Bat_Descarga_kW': vec_dis,
        'Bat_Carga_kW': vec_ch,
        'SoC_kWh': vec_soc_kwh,
        'SoC_Pct': vec_soc_pct,   # <--- NOVA COLUNA
        'Tensao_Max': df_v_real.drop('Hora', axis=1).max(axis=1),
        'Tensao_Min': df_v_real.drop('Hora', axis=1).replace(0, np.nan).min(axis=1).fillna(0)
    })
    
    try:
        with pd.ExcelWriter(excel_path) as writer:
            df_balanco.to_excel(writer, sheet_name="BALANCO_SISTEMA", index=False)
            
            rows_der = [{"Barra": b, "kW": d['cap_kw'], "kWh_Bat": d.get('cap_kwh', 0)} for b, d in decisions.items()]
            pd.DataFrame(rows_der).to_excel(writer, sheet_name="DERs_INSTALADOS", index=False)
            
            full_data['voltages'].to_excel(writer, sheet_name="LOG_TENSOES", index=False)
    except Exception as e: print(f"Erro Excel: {e}")

    # --- 2. GRÁFICO DE DESPACHO (CORRIGIDO) ---
    print("   -> Gerando Gráfico de Despacho...")
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # --- 2. GRÁFICO DE DESPACHO (EM INGLÊS + PENETRAÇÃO) ---
    print("   -> Generating Dispatch Chart (English)...")
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # A. Geração Positiva (Stacked)
    # Ordem: PV (Base) -> Wind -> Bateria -> Rede (Topo)
    p1 = ax1.bar(ts, vec_pv, label='Solar (PV)', color='#f1c40f', alpha=0.9, edgecolor='black', linewidth=0.5)
    
    bot_wind = vec_pv
    p2 = ax1.bar(ts, vec_wind, bottom=bot_wind, label='Wind', color='#27ae60', alpha=0.9, edgecolor='black', linewidth=0.5)
    
    bot_dis = bot_wind + vec_wind
    p3 = ax1.bar(ts, vec_dis, bottom=bot_dis, label='Battery (Discharge)', color='#e67e22', alpha=0.9, edgecolor='black', linewidth=0.5)
    
    bot_grid = bot_dis + vec_dis
    p4 = ax1.bar(ts, vec_grid, bottom=bot_grid, label='Grid (Import)', color='#7f8c8d', alpha=0.5, hatch='..', edgecolor='black', linewidth=0.5)
    
    # B. Consumo Bateria (Negativo)
    p5 = ax1.bar(ts, -vec_ch, label='Battery (Charge)', color='#c0392b', alpha=0.8, hatch='///', edgecolor='black', linewidth=0.5)
    
    # C. Linha de Carga
    p6 = ax1.plot(ts, vec_load, color='black', linestyle='-', linewidth=3, marker='o', markersize=6, label='Total Load', zorder=10)
    
    # Configuração Eixo 1
    ax1.axhline(0, color='black', linewidth=1)
    ax1.set_xlabel('Hour of Day', fontsize=12)
    ax1.set_ylabel('Power (kW)', fontsize=12)
    ax1.set_title(f'Optimal Energy Dispatch (DER Penetration: {penetration_val:.2f}%)', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # D. Eixo Secundário: SoC (%)
    ax2 = ax1.twinx()
    
    # Plot da Porcentagem
    p7 = ax2.plot(ts, vec_soc_pct, color='blue', linestyle='--', linewidth=2, marker='s', markersize=8, label='SoC Bateria (%)')
    
    # CORREÇÃO: Rótulo e Escala
    ax2.set_ylabel('State of Charge (%)', color='blue', fontsize=12) 
    ax2.tick_params(axis='y', labelcolor='blue')
    
    # Fixar limite entre 0 e 110% para visualização clara (evita escalas estranhas como 0-5000)
    ax2.set_ylim(0, 110)
    
    # E. Legenda Unificada
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.10), ncol=4, shadow=True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(CFG["OUTPUT_DIR"], "Grafico_Despacho_Detalhado.png"))

    # --- 3. GRÁFICO DE TENSÃO ---
    print("   -> Gerando Gráfico de Tensão...")
    plt.figure(figsize=(14, 6))
    plt.plot(ts, df_balanco['Tensao_Max'], 'r-o', label='Máxima (Real)')
    plt.plot(ts, df_balanco['Tensao_Min'], 'b-o', label='Mínima (Real)')
    plt.axhline(CFG['V_MAX'], color='k', ls='--', label='Limite Sup')
    plt.axhline(CFG['V_MIN'], color='k', ls='--', label='Limite Inf')
    plt.fill_between(ts, CFG['V_MIN'], CFG['V_MAX'], color='green', alpha=0.1, label='Faixa Segura')
    
    plt.title("Monitoramento de Tensão (OpenDSS)")
    plt.ylabel("Tensão (p.u.)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(CFG["OUTPUT_DIR"], "Grafico_Tensao_Real.png"))
    
    # ==========================================================================
    # NOVO: GRÁFICO DE DISTRIBUIÇÃO DE DER POR ZONA (CLUSTERING)
    # ==========================================================================
    print("   -> Gerando Gráfico de Distribuição DER por Zona (K-Means)...")
    
    # Inicializa contadores
    zone_capacity = {'Urban': 0.0, 'Mixed': 0.0, 'Rural': 0.0}
    
    # Soma a capacidade instalada (kW) em cada zona
    total_installed = 0
    for bus, data in decisions.items():
        kw = data['cap_kw']
        if kw > 0:
            # Pega a zona da barra (se não achar, assume Mixed por segurança)
            zone = zone_map.get(bus, 'Mixed')
            zone_capacity[zone] += kw
            total_installed += kw
            
    # Prepara dados para o gráfico
    labels = []
    sizes = []
    colors = []
    color_map = {'Urban': '#3498db', 'Mixed': '#f1c40f', 'Rural': '#2ecc71'} # Azul, Amarelo, Verde
    
    for zone in ['Urban', 'Mixed', 'Rural']:
        if zone_capacity[zone] > 0:
            labels.append(f"{zone}\n({zone_capacity[zone]:.1f} kW)")
            sizes.append(zone_capacity[zone])
            colors.append(color_map[zone])

    if total_installed > 0:
        plt.figure(figsize=(8, 8))
        # Gráfico de Rosca (Donut Chart)
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', 
                startangle=140, pctdistance=0.85, textprops={'fontsize': 12})
        
        # Círculo branco no meio
        centre_circle = plt.Circle((0,0),0.70,fc='white')
        fig = plt.gcf()
        fig.gca().add_artist(centre_circle)
        
        plt.title(f"Distributed Energy Resource by Zone\n(Total: {total_installed:.1f} kW)", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(CFG["OUTPUT_DIR"], "Grafico_Distribuicao_Zonal.png"))
    else:
        print("   [AVISO] Nenhum DER instalado, pulando gráfico de zonas.")

    if decisions:
        # price_profile não é mais estritamente necessário aqui, mas mantemos assinatura
        price_profile = dash_data['ts_data']['Price']
            
        # --- ALTERAÇÃO AQUI: Passamos dash_data como argumento ---
        df_financial = calculate_financial_composition(decisions, df_buses, price_profile, dash_data=dash_data)
            
        if not df_financial.empty:
            plot_financial_composition(df_financial)
    else:
        print("   [AVISO] Sem DERs, pulando análise financeira.")

    # --- 4. LOG FORENSE DETALHADO (SUBSTITUIR ESTE BLOCO) ---
    print("   -> Gerando Log Forense Completo...")
    txt_path = os.path.join(CFG["OUTPUT_DIR"], "Log_Violacoes_Resumo.txt")
    
    # Recuperar cargas nodais (p.u.) e converter para kW
    nodal_loads_pu = dash_data.get('nodal_loads', {})
    
    viol_count = 0
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=========================================================\n")
        f.write("          RELATÓRIO DETALHADO DE VIOLAÇÕES\n")
        f.write("=========================================================\n")
        f.write(f"Limites: {CFG['V_MIN']} - {CFG['V_MAX']} p.u.\n")
        f.write(f"Nota: 'Saldo' positivo = Injeção na Rede. Negativo = Compra.\n\n")
        
        for t in ts:
            row_real = df_v_real.iloc[t]
            # Tenta pegar a tensão otimizada (se disponível)
            row_opt = df_v_opt.iloc[t] if (df_v_opt is not None and t < len(df_v_opt)) else None
            
            # Identificar Barras com Violação
            violators = []
            cols = [c for c in df_v_real.columns if c != 'Hora']
            
            for bus in cols:
                v_real = row_real[bus]
                # Checa se estourou limite (com tolerância mínima)
                if v_real > CFG['V_MAX'] + 0.0001 or (v_real < CFG['V_MIN'] - 0.0001 and v_real > 0.1):
                    v_calc = row_opt[bus] if (row_opt is not None and bus in row_opt) else 0.0
                    violators.append((bus, v_real, v_calc))
            
            if violators:
                viol_count += 1
                f.write(f">>> HORA {t:02d}:00 - {len(violators)} violações detectadas.\n")
                f.write("-" * 80 + "\n")
                
                # Ordenar por gravidade
                violators.sort(key=lambda x: max(x[1]-CFG['V_MAX'], CFG['V_MIN']-x[1]), reverse=True)
                
                for bus, v_real, v_calc in violators:
                    # Recupera Carga da Barra nesta hora (kW)
                    # L_P é p.u., multiplicar por S_BASE*1000
                    load_kw = 0.0
                    if bus in nodal_loads_pu:
                        load_kw = nodal_loads_pu[bus][t] * CFG["S_BASE"] * 1000
                    
                    f.write(f"   [BARRA: {bus}]\n")
                    f.write(f"      Tensão Real (OpenDSS): {v_real:.4f} p.u.\n")
                    f.write(f"      Tensão Calc (Otimiz):  {v_calc:.4f} p.u. (Erro: {v_real-v_calc:+.4f})\n")
                    f.write(f"      Carga Consumida:       {load_kw:.2f} kW\n")
                    
                    # Detalhes do DER (se existir)
                    if bus in decisions:
                        d = decisions[bus]
                        dts = d['ts'] # Pega o dicionário detalhado criado no optimizer
                        
                        pv = dts['pv'][t]
                        wd = dts['wind'][t]
                        dis = dts['dis'][t]
                        ch = dts['ch'][t]
                        soc = dts['soc'][t]
                        cap = d['cap_kw']
                        
                        injecao_der = pv + wd + dis - ch
                        saldo_rede = injecao_der - load_kw # O que vai para o transformador
                        
                        f.write(f"      --- DER Instalado ({cap:.2f} kW) ---\n")
                        f.write(f"      Geração PV:      {pv:.2f} kW\n")
                        f.write(f"      Geração Wind:    {wd:.2f} kW\n")
                        f.write(f"      Bateria:         {dis:.2f} kW (Descarga) | {ch:.2f} kW (Carga)\n")
                        f.write(f"      Estado (SoC):    {soc:.2f} kWh\n")
                        f.write(f"      Injeção Líquida: {injecao_der:.2f} kW\n")
                        f.write(f"      SALDO FINAL:     {saldo_rede:+.2f} kW (Import/Export)\n")
                        
                        # Diagnóstico Rápido
                        if v_real > CFG['V_MAX']:
                            if saldo_rede > 0: f.write("      [CAUSA]: Excesso de injeção local (Exportação alta).\n")
                            else: f.write("      [CAUSA]: Tensão alta vinda da rede (não é culpa deste DER).\n")
                        
                    else:
                        f.write(f"      --- Sem DER Instalado ---\n")
                        f.write(f"      [CAUSA]: Problema sistêmico ou vizinho exportando muito.\n")
                    
                    f.write("\n")
                f.write("=" * 80 + "\n\n")

        if viol_count == 0:
            f.write("SUCESSO: Nenhuma violação de tensão detectada em todo o período.\n")
            
    print(f"   -> Log Detalhado salvo: {txt_path}")