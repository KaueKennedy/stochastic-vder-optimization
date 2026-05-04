from logging import config
import os
import re
import math
import shutil
import tempfile
import datetime
import pandas as pd
import py_dss_interface
import numpy as np

CONFIG = {
    "output_xlsx_path": r"c:\Users\KKCOD\OneDrive - Université Laval\Recherche\codes\code2\output\Optimization_Exact_Output.xlsx",
    "sbase_mva": 1.0,
    "default_frequency_hz": 60.0,
    "voltage_limits_pu": {
        "min": 0.95,
        "max": 1.05
    },
    "thermal_limits": {
        "use_norm_amps": True,
        "fallback_to_emerg_amps": True,
        "allow_missing_limits": True
    },
    "profiles": {
        "time_step_hours": 1.0,
        "load_key_candidates": ["BusProfiles", "loadcurve", "load"],
        "solar_key_candidates": ["PV", "solar"],
        "wind_key_candidates": ["Wind", "wind"],
        "price_key_candidates": ["LMP", "price"]
    },
    "optimization": {
        "objective": "stochastic_dispatch",
        "allow_curtailment": True,
        "allow_storage": True,
        "allow_generation_creation": True,
        "allow_network_parameter_estimation": False,
        "der_penetration_limit_percent": 35.0
    },
    "economic_parameters": {
        "solar_pv": {
            "capex_usd_kw": 1200.0,
            "opex_usd_kw_yr": 15.0,
            "max_install_kw_per_bus": 500.0
        },
        "wind": {
            "capex_usd_kw": 1500.0,
            "opex_usd_kw_yr": 20.0,
            "max_install_kw_per_bus": 1000.0
        },
        "battery_storage": {
            "capex_usd_kwh": 350.0,
            "opex_usd_kwh_yr": 5.0,
            "efficiency_round_trip": 0.85,
            "soc_min": 0.20,
            "soc_max": 0.95,
            "max_install_kwh_per_bus": 2000.0
        }
    }
}

def load_network_from_master(master_dss_path=None, output_xlsx_path=None):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))

    if master_dss_path is None:
        master_dss_path = os.path.join(
            project_root,
            "Iowa_Distribution_Test_Systems",
            "OpenDSS Model",
            "OpenDSS Model",
            "Master.dss",
        )

        if output_xlsx_path is None:
            output_xlsx_path = os.path.join(
                current_dir,
                "parameters",
                "Network_Data_Imported.xlsx",
            )

        master_dss_path = os.path.abspath(master_dss_path)
        output_xlsx_path = os.path.abspath(output_xlsx_path)
        log_path = os.path.join(
            current_dir,
            "output",
            "Network_Data_Imported_debug_log.txt",
        )

    log_lines = []

    def log(message):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}"
        log_lines.append(line)
        print(line)

    def flush_log():
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("\n".join(log_lines))

    def contains_non_ascii(text):
        try:
            text.encode("ascii")
            return False
        except UnicodeEncodeError:
            return True

    def safe_float(value, default=None):
        try:
            if value is None or value == "":
                return default
            return float(value)
        except Exception:
            return default

    def safe_int(value, default=None):
        try:
            if value is None or value == "":
                return default
            return int(float(value))
        except Exception:
            return default

    def safe_str(value, default=""):
        try:
            if value is None:
                return default
            return str(value)
        except Exception:
            return default

    def safe_list(value):
        try:
            if value is None:
                return []
            return list(value)
        except Exception:
            return []

    def list_to_text(value):
        try:
            if value is None:
                return ""
            if isinstance(value, (list, tuple)):
                return ",".join(str(x) for x in value)
            return ",".join(str(x) for x in list(value))
        except Exception:
            return safe_str(value, "")

    def clean_bus_name(bus_name):
        if bus_name is None:
            return ""
        return str(bus_name).split(".")[0].strip()

    def normalize_bus_name(bus_name):
        if bus_name is None:
            return ""
        name = str(bus_name).strip().lower()
        name = name.split(".")[0]
        name = name.replace("-", "_").replace(" ", "_")
        name = re.sub(r"__+", "_", name)
        return name

    def normalize_secondary_bus_name(bus_name):
        name = normalize_bus_name(bus_name)
        name = name.replace("tbus", "t_bus")
        if re.match(r"^t_bus\d+l$", name):
            name = name[:-1] + "_l"
        return name

    def infer_primary_bus_from_secondary(bus_name):
        name = normalize_secondary_bus_name(bus_name)
        match = re.match(r"^t_bus(\d+)_l$", name)
        if match:
            return f"bus{match.group(1)}"
        return ""

    def voltage_magnitudes_from_complex_array(values):
        magnitudes = []
        vals = safe_list(values)
        for i in range(0, len(vals), 2):
            re_part = safe_float(vals[i], 0.0)
            im_part = safe_float(vals[i + 1], 0.0) if i + 1 < len(vals) else 0.0
            magnitudes.append(math.sqrt(re_part ** 2 + im_part ** 2))
        return magnitudes

    def safe_call(getter, default=None, context=""):
        try:
            return getter()
        except Exception as e:
            log(f"ERROR | {context} | {type(e).__name__} | {e}")
            return default

    log("Starting load_network_from_master()")
    log(f"current_dir = {current_dir}")
    log(f"project_root = {project_root}")
    log(f"master_dss_path = {master_dss_path}")
    log(f"output_xlsx_path = {output_xlsx_path}")
    log(f"log_path = {log_path}")

    compile_master_path = master_dss_path
    temp_workspace = None

    try:
        if not os.path.exists(master_dss_path):
            raise FileNotFoundError(f"Master DSS file not found: {master_dss_path}")

        if contains_non_ascii(master_dss_path):
            log("Non-ASCII path detected, creating temporary ASCII copy")
            original_model_dir = os.path.dirname(master_dss_path)
            temp_workspace = tempfile.mkdtemp(prefix="dss_ascii_")
            ascii_model_dir = os.path.join(temp_workspace, "OpenDSS_Model")
            shutil.copytree(original_model_dir, ascii_model_dir, dirs_exist_ok=True)
            compile_master_path = os.path.join(ascii_model_dir, os.path.basename(master_dss_path))
            log(f"compile_master_path remapped to {compile_master_path}")

        dss = py_dss_interface.DSS()
        log("DSS object created")

        dss.text(f'Compile "{compile_master_path}"')
        log("Compile command sent")

        circuit_name = safe_str(dss.circuit.name)
        log(f"circuit.name = {circuit_name}")

        if not circuit_name:
            raise RuntimeError("Compile completed but no active circuit was created.")

        num_buses = safe_int(dss.circuit.num_buses, 0)
        num_nodes = safe_int(dss.circuit.num_nodes, 0)
        num_elements = safe_int(dss.circuit.num_ckt_elements, 0)

        log(f"num_buses = {num_buses}")
        log(f"num_nodes = {num_nodes}")
        log(f"num_circuit_elements = {num_elements}")

        all_bus_names = safe_list(dss.circuit.buses_names)
        all_element_names = safe_list(dss.circuit.elements_names)
        all_node_names = safe_list(dss.circuit.nodes_names)

        log(f"len(all_bus_names) = {len(all_bus_names)}")
        log(f"len(all_element_names) = {len(all_element_names)}")
        log(f"len(all_node_names) = {len(all_node_names)}")
        log(f"first_10_buses = {all_bus_names[:10]}")
        log(f"first_10_elements = {all_element_names[:10]}")

        line_names = safe_call(lambda: list(dss.lines.names), default=[], context="Lines:names")
        transformer_names = safe_call(lambda: list(dss.transformers.names), default=[], context="Transformers:names")
        load_names = safe_call(lambda: list(dss.loads.names), default=[], context="Loads:names")

        log(f"len(line_names) = {len(line_names)}")
        log(f"len(transformer_names) = {len(transformer_names)}")
        log(f"len(load_names) = {len(load_names)}")
        log(f"first_10_lines = {line_names[:10]}")
        log(f"first_10_transformers = {transformer_names[:10]}")
        log(f"first_10_loads = {load_names[:10]}")

        if len(all_bus_names) == 0:
            raise RuntimeError("No buses were found after compile.")
        if len(all_element_names) == 0:
            raise RuntimeError("No circuit elements were found after compile.")

        circuit_info = {
            "name": circuit_name,
            "num_buses": num_buses,
            "num_nodes": num_nodes,
            "num_circuit_elements": num_elements,
        }

        buses_data = []
        log("Starting bus extraction")
        for idx, bus_name in enumerate(all_bus_names, start=1):
            dss.circuit.set_active_bus(bus_name)

            pu_voltages = safe_call(lambda: dss.bus.pu_voltages, default=[], context=f"Bus:{bus_name}:pu_voltages")
            magnitudes = voltage_magnitudes_from_complex_array(pu_voltages)

            row = {
                "BusName": normalize_secondary_bus_name(bus_name),
                "OriginalBusName": safe_str(bus_name),
                "BaseKV": safe_call(lambda: safe_float(dss.bus.kv_base), None, f"Bus:{bus_name}:kv_base"),
                "NumNodes": safe_call(lambda: safe_int(dss.bus.num_nodes), None, f"Bus:{bus_name}:num_nodes"),
                "Nodes": safe_call(lambda: list_to_text(dss.bus.nodes), "", f"Bus:{bus_name}:nodes"),
                "X": safe_call(lambda: safe_float(dss.bus.x), None, f"Bus:{bus_name}:x"),
                "Y": safe_call(lambda: safe_float(dss.bus.y), None, f"Bus:{bus_name}:y"),
                "Distance": safe_call(lambda: safe_float(dss.bus.distance), None, f"Bus:{bus_name}:distance"),
                "VoltagesPU": list_to_text(pu_voltages),
                "VoltageMagnitudesPU": list_to_text(magnitudes),
                "PrimaryBusReference": infer_primary_bus_from_secondary(bus_name),
                "Lambda": safe_call(lambda: safe_float(dss.bus.bus_lambda), None, f"Bus:{bus_name}:bus_lambda"),
                "NumCustomers": safe_call(lambda: safe_int(dss.bus.total_customers), None, f"Bus:{bus_name}:total_customers"),
                "NumInterrupts": safe_call(lambda: safe_float(dss.bus.interruptions_num), None, f"Bus:{bus_name}:interruptions_num"),
                "CustDuration": safe_call(lambda: safe_float(dss.bus.interruptions_avg_duration), None, f"Bus:{bus_name}:interruptions_avg_duration"),
                "TotalMiles": safe_call(lambda: safe_float(dss.bus.line_total_miles), None, f"Bus:{bus_name}:line_total_miles"),
                "SectionID": safe_call(lambda: safe_int(dss.bus.section_id), None, f"Bus:{bus_name}:section_id"),
            }
            buses_data.append(row)

            if idx <= 5:
                log(f"bus[{idx}] = {row['BusName']}")

        buses_df = pd.DataFrame(buses_data)

        # Atualiza coordenadas zeradas usando PrimaryBusReference
        if not buses_df.empty:
            coord_map = {}
            for _, row in buses_df.iterrows():
                ref_bus_name = normalize_bus_name(row["BusName"])
                x = safe_float(row["X"], 0.0)
                y = safe_float(row["Y"], 0.0)
                if abs(x) > 1e-9 or abs(y) > 1e-9:
                    coord_map[ref_bus_name] = (x, y)

            for idx, row in buses_df.iterrows():
                x = safe_float(row["X"], 0.0)
                y = safe_float(row["Y"], 0.0)
                if abs(x) <= 1e-9 and abs(y) <= 1e-9:
                    primary_ref = normalize_bus_name(row.get("PrimaryBusReference", ""))
                    if primary_ref and primary_ref in coord_map:
                        buses_df.at[idx, "X"] = coord_map[primary_ref][0]
                        buses_df.at[idx, "Y"] = coord_map[primary_ref][1]

        lines_data = []
        log("Starting line extraction")

        lines_data = []
        log("Starting line extraction")
        for idx, line_name in enumerate(line_names, start=1):
            try:
                dss.circuit.set_active_class("line")
            except Exception:
                pass

            try:
                dss.lines.name = line_name
                dss.circuit.set_active_element(f"Line.{line_name}")
            except Exception as e:
                log(f"ERROR | Line:{line_name}:activate | {type(e).__name__} | {e}")
                continue

            buses = safe_call(lambda: list(dss.cktelement.bus_names), default=[], context=f"Line:{line_name}:bus_names")
            length_value = safe_call(lambda: safe_float(dss.lines.length, 0.0), default=0.0, context=f"Line:{line_name}:length")
            r1_value = safe_call(lambda: safe_float(dss.lines.r1), None, f"Line:{line_name}:r1")
            x1_value = safe_call(lambda: safe_float(dss.lines.x1), None, f"Line:{line_name}:x1")
            r0_value = safe_call(lambda: safe_float(dss.lines.r0), None, f"Line:{line_name}:r0")
            x0_value = safe_call(lambda: safe_float(dss.lines.x0), None, f"Line:{line_name}:x0")

            row = {
                "Name": safe_str(line_name),
                "Bus1": normalize_secondary_bus_name(clean_bus_name(buses[0])) if len(buses) > 0 else "",
                "Bus2": normalize_secondary_bus_name(clean_bus_name(buses[1])) if len(buses) > 1 else "",
                "Bus1Full": buses[0] if len(buses) > 0 else "",
                "Bus2Full": buses[1] if len(buses) > 1 else "",
                "Phases": safe_call(lambda: safe_int(dss.lines.phases), None, f"Line:{line_name}:phases"),
                "Length": length_value,
                "Units": safe_call(lambda: safe_str(dss.lines.units), "", f"Line:{line_name}:units"),
                "LineCode": safe_call(lambda: safe_str(dss.lines.linecode), "", f"Line:{line_name}:linecode"),
                "Geometry": safe_call(lambda: safe_str(dss.lines.geometry), "", f"Line:{line_name}:geometry"),
                "Spacing": safe_call(lambda: safe_str(dss.lines.spacing), "", f"Line:{line_name}:spacing"),
                "NormAmps": safe_call(lambda: safe_float(dss.lines.norm_amps), None, f"Line:{line_name}:norm_amps"),
                "EmergAmps": safe_call(lambda: safe_float(dss.lines.emerg_amps), None, f"Line:{line_name}:emerg_amps"),
                "R1": r1_value,
                "X1": x1_value,
                "R0": r0_value,
                "X0": x0_value,
                "C1": safe_call(lambda: safe_float(dss.lines.c1), None, f"Line:{line_name}:c1"),
                "C0": safe_call(lambda: safe_float(dss.lines.c0), None, f"Line:{line_name}:c0"),
                "R1Total": (r1_value * length_value) if r1_value is not None else None,
                "X1Total": (x1_value * length_value) if x1_value is not None else None,
                "R0Total": (r0_value * length_value) if r0_value is not None else None,
                "X0Total": (x0_value * length_value) if x0_value is not None else None,
                "Enabled": safe_call(lambda: safe_int(dss.cktelement.enabled), None, f"Line:{line_name}:enabled"),
            }
            lines_data.append(row)

            if idx <= 5:
                log(f"line[{idx}] = {row['Name']}")

        lines_df = pd.DataFrame(lines_data)

        transformers_data = []
        log("Starting transformer extraction")
        for idx, transformer_name in enumerate(transformer_names, start=1):
            try:
                dss.circuit.set_active_class("transformer")
            except Exception:
                pass

            try:
                dss.transformers.name = transformer_name
                dss.circuit.set_active_element(f"Transformer.{transformer_name}")
            except Exception as e:
                log(f"ERROR | Transformer:{transformer_name}:activate | {type(e).__name__} | {e}")
                continue

            buses = safe_call(lambda: list(dss.cktelement.bus_names), default=[], context=f"Transformer:{transformer_name}:bus_names")
            num_windings = safe_call(lambda: safe_int(dss.transformers.num_windings, 0), default=0, context=f"Transformer:{transformer_name}:num_windings")

            row = {
                "Name": safe_str(transformer_name),
                "NumWindings": num_windings,
                "Phases": safe_call(lambda: safe_int(dss.cktelement.num_phases), None, f"Transformer:{transformer_name}:num_phases"),
                "Bus1": normalize_secondary_bus_name(clean_bus_name(buses[0])) if len(buses) > 0 else "",
                "Bus2": normalize_secondary_bus_name(clean_bus_name(buses[1])) if len(buses) > 1 else "",
                "Bus1Full": buses[0] if len(buses) > 0 else "",
                "Bus2Full": buses[1] if len(buses) > 1 else "",
                "Xhl": safe_call(lambda: safe_float(dss.transformers.xhl), None, f"Transformer:{transformer_name}:xhl"),
                "Xht": safe_call(lambda: safe_float(dss.transformers.xht), None, f"Transformer:{transformer_name}:xht"),
                "Xlt": safe_call(lambda: safe_float(dss.transformers.xlt), None, f"Transformer:{transformer_name}:xlt"),
                "IsDelta": safe_call(lambda: safe_int(dss.transformers.is_delta), None, f"Transformer:{transformer_name}:is_delta"),
                "MaxTap": safe_call(lambda: safe_float(dss.transformers.max_tap), None, f"Transformer:{transformer_name}:max_tap"),
                "MinTap": safe_call(lambda: safe_float(dss.transformers.min_tap), None, f"Transformer:{transformer_name}:min_tap"),
                "NumTaps": safe_call(lambda: safe_int(dss.transformers.num_taps), None, f"Transformer:{transformer_name}:num_taps"),
                "Tap": safe_call(lambda: safe_float(dss.transformers.tap), None, f"Transformer:{transformer_name}:tap"),
                "Enabled": safe_call(lambda: safe_int(dss.cktelement.enabled), None, f"Transformer:{transformer_name}:enabled"),
            }

            for wdg in range(1, num_windings + 1):
                try:
                    dss.transformers.wdg = wdg
                except Exception as e:
                    log(f"ERROR | Transformer:{transformer_name}:W{wdg}:set_wdg | {type(e).__name__} | {e}")
                    continue

                row[f"W{wdg}_KV"] = safe_call(lambda: safe_float(dss.transformers.kv), None, f"Transformer:{transformer_name}:W{wdg}:kv")
                row[f"W{wdg}_KVA"] = safe_call(lambda: safe_float(dss.transformers.kva), None, f"Transformer:{transformer_name}:W{wdg}:kva")
                row[f"W{wdg}_Tap"] = safe_call(lambda: safe_float(dss.transformers.tap), None, f"Transformer:{transformer_name}:W{wdg}:tap")
                row[f"W{wdg}_RPercent"] = safe_call(lambda: safe_float(dss.transformers.r), None, f"Transformer:{transformer_name}:W{wdg}:r")
                row[f"W{wdg}_Rneut"] = safe_call(lambda: safe_float(dss.transformers.r_neut), None, f"Transformer:{transformer_name}:W{wdg}:r_neut")
                row[f"W{wdg}_Xneut"] = safe_call(lambda: safe_float(dss.transformers.x_neut), None, f"Transformer:{transformer_name}:W{wdg}:x_neut")

            transformers_data.append(row)

            if idx <= 5:
                log(f"transformer[{idx}] = {row['Name']}")

        transformers_df = pd.DataFrame(transformers_data)

        loads_data = []
        log("Starting load extraction")
        for idx, load_name in enumerate(load_names, start=1):
            try:
                dss.circuit.set_active_class("load")
            except Exception:
                pass

            try:
                dss.loads.name = load_name
                dss.circuit.set_active_element(f"Load.{load_name}")
            except Exception as e:
                log(f"ERROR | Load:{load_name}:activate | {type(e).__name__} | {e}")
                continue

            buses = safe_call(lambda: list(dss.cktelement.bus_names), default=[], context=f"Load:{load_name}:bus_names")
            is_delta = safe_call(lambda: safe_int(dss.loads.is_delta, 0), default=0, context=f"Load:{load_name}:is_delta")

            row = {
                "Name": safe_str(load_name),
                "Bus": normalize_secondary_bus_name(clean_bus_name(buses[0])) if len(buses) > 0 else "",
                "BusFull": buses[0] if len(buses) > 0 else "",
                "Phases": safe_call(lambda: safe_int(dss.cktelement.num_phases), None, f"Load:{load_name}:num_phases"),
                "Conn": "delta" if is_delta == 1 else "wye",
                "Model": safe_call(lambda: safe_int(dss.loads.model), None, f"Load:{load_name}:model"),
                "KV": safe_call(lambda: safe_float(dss.loads.kv), None, f"Load:{load_name}:kv"),
                "KW": safe_call(lambda: safe_float(dss.loads.kw), None, f"Load:{load_name}:kw"),
                "Kvar": safe_call(lambda: safe_float(dss.loads.kvar), None, f"Load:{load_name}:kvar"),
                "KVA": safe_call(lambda: safe_float(dss.loads.kva), None, f"Load:{load_name}:kva"),
                "PF": safe_call(lambda: safe_float(dss.loads.pf), None, f"Load:{load_name}:pf"),
                "Daily": safe_call(lambda: safe_str(dss.loads.daily), "", f"Load:{load_name}:daily"),
                "Yearly": safe_call(lambda: safe_str(dss.loads.yearly), "", f"Load:{load_name}:yearly"),
                "Duty": safe_call(lambda: safe_str(dss.loads.duty), "", f"Load:{load_name}:duty"),
                "Growth": safe_call(lambda: safe_str(dss.loads.growth), "", f"Load:{load_name}:growth"),
                "Status": safe_call(lambda: safe_str(dss.loads.status), "", f"Load:{load_name}:status"),
                "Class": safe_call(lambda: safe_int(dss.loads.class_number), None, f"Load:{load_name}:class_number"),
                "VminPU": safe_call(lambda: safe_float(dss.loads.vmin_pu), None, f"Load:{load_name}:vmin_pu"),
                "VmaxPU": safe_call(lambda: safe_float(dss.loads.vmax_pu), None, f"Load:{load_name}:vmax_pu"),
                "VminNorm": safe_call(lambda: safe_float(dss.loads.vmin_norm), None, f"Load:{load_name}:vmin_norm"),
                "VminEmerg": safe_call(lambda: safe_float(dss.loads.vmin_emerg), None, f"Load:{load_name}:vmin_emerg"),
                "NumCust": safe_call(lambda: safe_int(dss.loads.num_cust), None, f"Load:{load_name}:num_cust"),
                "ZIPV": safe_call(lambda: list_to_text(dss.loads.zipv), "", f"Load:{load_name}:zipv"),
                "Enabled": safe_call(lambda: safe_int(dss.cktelement.enabled), None, f"Load:{load_name}:enabled"),
            }
            loads_data.append(row)

            if idx <= 5:
                log(f"load[{idx}] = {row['Name']}")

        loads_df = pd.DataFrame(loads_data)

        summary_df = pd.DataFrame([
            {"Item": "CircuitName", "Value": circuit_info["name"]},
            {"Item": "OriginalMasterDSSPath", "Value": master_dss_path},
            {"Item": "CompiledMasterDSSPath", "Value": compile_master_path},
            {"Item": "OutputXLSXPath", "Value": output_xlsx_path},
            {"Item": "LogPath", "Value": log_path},
            {"Item": "NumBuses", "Value": len(buses_df)},
            {"Item": "NumLines", "Value": len(lines_df)},
            {"Item": "NumTransformers", "Value": len(transformers_df)},
            {"Item": "NumLoads", "Value": len(loads_df)},
            {"Item": "NumCircuitElements", "Value": len(all_element_names)},
            {"Item": "NumNodeNames", "Value": len(all_node_names)},
        ])

        elements_df = pd.DataFrame({"ElementName": all_element_names})
        busnames_df = pd.DataFrame({"BusName": all_bus_names})
        nodenames_df = pd.DataFrame({"NodeName": all_node_names})

        os.makedirs(os.path.dirname(output_xlsx_path), exist_ok=True)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        log("Saving Excel workbook")

        with pd.ExcelWriter(output_xlsx_path, engine="openpyxl") as writer:
            summary_df.to_excel(writer, sheet_name="Summary", index=False)
            buses_df.to_excel(writer, sheet_name="Buses", index=False)
            lines_df.to_excel(writer, sheet_name="Lines", index=False)
            transformers_df.to_excel(writer, sheet_name="Transformers", index=False)
            loads_df.to_excel(writer, sheet_name="Loads", index=False)
            elements_df.to_excel(writer, sheet_name="AllElements", index=False)
            busnames_df.to_excel(writer, sheet_name="AllBuses", index=False)
            nodenames_df.to_excel(writer, sheet_name="AllNodes", index=False)

        log("Excel workbook saved successfully")

        imported_data = {
            "circuit_info": circuit_info,
            "master_dss_path": master_dss_path,
            "compiled_master_dss_path": compile_master_path,
            "output_xlsx_path": output_xlsx_path,
            "log_path": log_path,
            "buses": buses_df,
            "lines": lines_df,
            "transformers": transformers_df,
            "loads": loads_df,
            "all_elements": elements_df,
            "all_buses": busnames_df,
            "all_nodes": nodenames_df,
            "summary": summary_df,
        }

        log("Returning imported data")
        flush_log()
        return imported_data

    except Exception as e:
        log(f"FATAL ERROR | {type(e).__name__} | {e}")
        flush_log()
        raise

def load_or_create_profiles(imported_data, profiles_xlsx_path=None, year=2025):
    import os
    import hashlib
    import datetime
    import numpy as np
    import pandas as pd

    current_dir = os.path.dirname(os.path.abspath(__file__))
    parameters_directory = os.path.join(current_dir, "parameters")

    if profiles_xlsx_path is None:
        profiles_xlsx_path = os.path.join(parameters_directory, "profiles.xlsx")

    profiles_xlsx_path = os.path.abspath(profiles_xlsx_path)
    os.makedirs(os.path.dirname(profiles_xlsx_path), exist_ok=True)

    if not isinstance(imported_data, dict):
        raise TypeError("imported_data must be a dictionary.")

    if "buses" not in imported_data:
        raise KeyError("imported_data must contain the key 'buses'.")

    log_path = imported_data.get(
        "log_path",
        os.path.join(current_dir, "output", "Network_Data_Imported_debug_log.txt")
    )

    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def log(message):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}"
        print(line)
        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(line + "\n")

    log("Starting load_or_create_profiles()")
    log(f"profiles_xlsx_path = {profiles_xlsx_path}")
    log(f"year = {year}")

    buses_dataframe = imported_data["buses"]

    if buses_dataframe is None or buses_dataframe.empty:
        raise ValueError("imported_data['buses'] is empty.")

    if "BusName" not in buses_dataframe.columns:
        raise ValueError("The buses DataFrame must contain a 'BusName' column.")

    log(f"received_buses_rows = {len(buses_dataframe)}")
    log(f"buses_columns = {list(buses_dataframe.columns)}")

    if os.path.exists(profiles_xlsx_path):
        try:
            excel_file = pd.ExcelFile(profiles_xlsx_path)
            required_sheet_names = {"BusProfiles", "PV", "Wind", "LMP", "Metadata"}

            log(f"existing_profiles_file_detected = {profiles_xlsx_path}")
            log(f"existing_sheet_names = {excel_file.sheet_names}")

            if required_sheet_names.issubset(set(excel_file.sheet_names)):
                profiles_data = {
                    "BusProfiles": pd.read_excel(profiles_xlsx_path, sheet_name="BusProfiles"),
                    "PV": pd.read_excel(profiles_xlsx_path, sheet_name="PV"),
                    "Wind": pd.read_excel(profiles_xlsx_path, sheet_name="Wind"),
                    "LMP": pd.read_excel(profiles_xlsx_path, sheet_name="LMP"),
                    "Metadata": pd.read_excel(profiles_xlsx_path, sheet_name="Metadata"),
                    "file_path": profiles_xlsx_path,
                    "created_new_file": False,
                }

                log(f"loaded_existing_profiles_successfully = True")
                log(f"BusProfiles_rows = {len(profiles_data['BusProfiles'])}")
                log(f"PV_rows = {len(profiles_data['PV'])}")
                log(f"Wind_rows = {len(profiles_data['Wind'])}")
                log(f"LMP_rows = {len(profiles_data['LMP'])}")
                log("Returning existing profiles data")
                return profiles_data
            else:
                missing_sheets = sorted(list(required_sheet_names - set(excel_file.sheet_names)))
                log(f"existing_profiles_file_incomplete = True")
                log(f"missing_sheets = {missing_sheets}")
                log("Recreating profiles.xlsx from imported_data")

        except Exception as exception:
            log(f"ERROR | reading existing profiles.xlsx | {type(exception).__name__} | {exception}")
            log("Recreating profiles.xlsx from imported_data")

    buses_dataframe = buses_dataframe.copy()
    buses_dataframe["BusName"] = buses_dataframe["BusName"].astype(str).str.strip()
    buses_dataframe = buses_dataframe.dropna(subset=["BusName"])
    buses_dataframe = buses_dataframe.drop_duplicates(subset=["BusName"]).reset_index(drop=True)

    log(f"cleaned_buses_rows = {len(buses_dataframe)}")

    def generate_stable_seed_from_name(name):
        digest = hashlib.md5(name.encode("utf-8")).hexdigest()
        return int(digest[:8], 16)

    def normalize_numeric_series(series, default_value=0.5):
        numeric_series = pd.to_numeric(series, errors="coerce")
        if numeric_series.notna().sum() == 0:
            return pd.Series(np.full(len(series), default_value), index=series.index)
        minimum_value = numeric_series.min()
        maximum_value = numeric_series.max()
        if pd.isna(minimum_value) or pd.isna(maximum_value) or maximum_value == minimum_value:
            return pd.Series(np.full(len(series), default_value), index=series.index)
        return (numeric_series - minimum_value) / (maximum_value - minimum_value)

    if "X" in buses_dataframe.columns:
        normalized_x = normalize_numeric_series(buses_dataframe["X"])
    else:
        normalized_x = pd.Series(np.full(len(buses_dataframe), 0.5), index=buses_dataframe.index)

    if "Y" in buses_dataframe.columns:
        normalized_y = normalize_numeric_series(buses_dataframe["Y"])
    else:
        normalized_y = pd.Series(np.full(len(buses_dataframe), 0.5), index=buses_dataframe.index)

    if "Distance" in buses_dataframe.columns:
        normalized_distance = normalize_numeric_series(buses_dataframe["Distance"])
    else:
        normalized_distance = pd.Series(np.full(len(buses_dataframe), 0.5), index=buses_dataframe.index)

    bus_profiles_rows = []

    for row_index, row in buses_dataframe.iterrows():
        bus_name = row["BusName"]
        random_seed = generate_stable_seed_from_name(f"{bus_name}_{year}")
        random_generator = np.random.default_rng(random_seed)

        normalized_x_value = float(normalized_x.iloc[row_index])
        normalized_y_value = float(normalized_y.iloc[row_index])
        normalized_distance_value = float(normalized_distance.iloc[row_index])

        if normalized_x_value < 0.33:
            geographic_cluster = "West"
        elif normalized_x_value < 0.66:
            geographic_cluster = "Central"
        else:
            geographic_cluster = "East"

        bus_name_lowercase = bus_name.lower()
        if bus_name_lowercase.startswith("bus1") or bus_name_lowercase.startswith("100"):
            feeder_group = "Feeder_A"
        elif bus_name_lowercase.startswith("bus2") or bus_name_lowercase.startswith("200"):
            feeder_group = "Feeder_B"
        elif bus_name_lowercase.startswith("bus3") or bus_name_lowercase.startswith("300"):
            feeder_group = "Feeder_C"
        else:
            feeder_group = "Unknown"

        loss_factor = 1.00 + 0.015 * normalized_distance_value + random_generator.normal(0.0, 0.003)
        loss_factor = float(np.clip(loss_factor, 1.000, 1.035))

        price_multiplier = 1.00 + 0.020 * normalized_distance_value + random_generator.normal(0.0, 0.005)
        price_multiplier = float(np.clip(price_multiplier, 0.98, 1.06))

        price_adder_usd_per_mwh = 1.5 + 8.0 * normalized_distance_value + random_generator.normal(0.0, 0.8)
        price_adder_usd_per_mwh = float(np.clip(price_adder_usd_per_mwh, 0.0, 12.0))

        load_scale = 0.92 + 0.18 * normalized_y_value + random_generator.normal(0.0, 0.035)
        load_scale = float(np.clip(load_scale, 0.85, 1.20))

        load_peak_shift_hours = int(
            np.clip(
                np.round(random_generator.normal(loc=(normalized_y_value - 0.5) * 1.5, scale=0.8)),
                -2,
                2,
            )
        )

        solar_scale = 0.85 + 0.22 * normalized_x_value - 0.08 * normalized_distance_value + random_generator.normal(0.0, 0.04)
        solar_scale = float(np.clip(solar_scale, 0.70, 1.15))

        solar_cloud_factor = 0.90 + 0.10 * np.sin(2 * np.pi * normalized_x_value) + random_generator.normal(0.0, 0.03)
        solar_cloud_factor = float(np.clip(solar_cloud_factor, 0.75, 1.05))

        solar_cloud_shift_hours = int(
            np.clip(
                np.round(random_generator.normal(loc=(normalized_x_value - 0.5) * 2.0, scale=1.0)),
                -2,
                2,
            )
        )

        wind_scale = 0.80 + 0.25 * normalized_distance_value + 0.10 * (1.0 - normalized_y_value) + random_generator.normal(0.0, 0.05)
        wind_scale = float(np.clip(wind_scale, 0.65, 1.20))

        wind_phase_shift_hours = int(
            np.clip(
                np.round(random_generator.normal(loc=(normalized_distance_value - 0.5) * 3.0, scale=1.2)),
                -3,
                3,
            )
        )

        local_variability = 0.03 + 0.04 * random_generator.random()
        local_variability = float(np.clip(local_variability, 0.03, 0.07))

        bus_profiles_rows.append({
            "BusName": bus_name,
            "Year": int(year),
            "FeederGroup": feeder_group,
            "GeographicCluster": geographic_cluster,
            "NormalizedX": round(normalized_x_value, 6),
            "NormalizedY": round(normalized_y_value, 6),
            "NormalizedDistance": round(normalized_distance_value, 6),
            "LossFactor": round(loss_factor, 6),
            "PriceMultiplier": round(price_multiplier, 6),
            "PriceAdderUSDperMWh": round(price_adder_usd_per_mwh, 6),
            "LoadScale": round(load_scale, 6),
            "LoadPeakShiftHours": int(load_peak_shift_hours),
            "SolarScale": round(solar_scale, 6),
            "SolarCloudFactor": round(solar_cloud_factor, 6),
            "SolarCloudShiftHours": int(solar_cloud_shift_hours),
            "WindScale": round(wind_scale, 6),
            "WindPhaseShiftHours": int(wind_phase_shift_hours),
            "LocalVariability": round(local_variability, 6),
            "RandomSeedBus": int(random_seed),
        })

    bus_profiles_dataframe = pd.DataFrame(bus_profiles_rows).sort_values("BusName").reset_index(drop=True)

    log(f"generated_bus_profiles_rows = {len(bus_profiles_dataframe)}")
    log(f"first_5_bus_profiles = {bus_profiles_dataframe['BusName'].head(5).tolist()}")

    seasonal_factors = {
        1: {"PV": 0.70, "Wind": 1.12, "LMP": 1.10},
        2: {"PV": 0.78, "Wind": 1.08, "LMP": 1.06},
        3: {"PV": 0.92, "Wind": 0.98, "LMP": 1.00},
        4: {"PV": 1.02, "Wind": 0.95, "LMP": 0.97},
        5: {"PV": 1.10, "Wind": 0.92, "LMP": 0.95},
        6: {"PV": 1.18, "Wind": 0.90, "LMP": 0.96},
        7: {"PV": 1.20, "Wind": 0.88, "LMP": 1.00},
        8: {"PV": 1.12, "Wind": 0.90, "LMP": 1.02},
        9: {"PV": 1.00, "Wind": 0.95, "LMP": 1.01},
        10: {"PV": 0.90, "Wind": 1.00, "LMP": 1.00},
        11: {"PV": 0.78, "Wind": 1.06, "LMP": 1.05},
        12: {"PV": 0.75, "Wind": 1.10, "LMP": 1.08},
    }

    annual_means = {"PV": 0.35, "Wind": 0.42, "LMP": 55.0}
    daily_random_limits = {"PV": 0.06, "Wind": 0.10, "LMP": 0.08}

    day_night_factors = {
        "PV": {
            0: 0.00, 1: 0.00, 2: 0.00, 3: 0.00, 4: 0.00, 5: 0.05,
            6: 0.15, 7: 0.35, 8: 0.55, 9: 0.72, 10: 0.85, 11: 0.95,
            12: 1.00, 13: 0.96, 14: 0.88, 15: 0.75, 16: 0.58, 17: 0.35,
            18: 0.12, 19: 0.02, 20: 0.00, 21: 0.00, 22: 0.00, 23: 0.00,
        },
        "Wind": {
            0: 1.04, 1: 1.05, 2: 1.06, 3: 1.05, 4: 1.03, 5: 1.00,
            6: 0.98, 7: 0.96, 8: 0.94, 9: 0.93, 10: 0.92, 11: 0.93,
            12: 0.95, 13: 0.97, 14: 1.00, 15: 1.03, 16: 1.05, 17: 1.06,
            18: 1.07, 19: 1.08, 20: 1.08, 21: 1.07, 22: 1.06, 23: 1.05,
        },
        "LMP": {
            0: 0.92, 1: 0.90, 2: 0.89, 3: 0.88, 4: 0.90, 5: 0.96,
            6: 1.05, 7: 1.12, 8: 1.18, 9: 1.14, 10: 1.08, 11: 1.02,
            12: 0.99, 13: 0.98, 14: 0.99, 15: 1.03, 16: 1.08, 17: 1.16,
            18: 1.24, 19: 1.28, 20: 1.20, 21: 1.10, 22: 1.00, 23: 0.95,
        },
    }

    timestamp_index = pd.date_range(
        start=f"{year}-01-01 00:00:00",
        end=f"{year}-12-31 23:00:00",
        freq="h",
    )

    unique_days = pd.Series(pd.Series(timestamp_index.date).astype(str).unique())

    photovoltaic_daily_random_factors = {}
    wind_daily_random_factors = {}
    locational_marginal_price_daily_random_factors = {}

    for day_index, day_value in enumerate(unique_days):
        photovoltaic_daily_random_factors[day_value] = 1.0 + (((day_index * 37) % 1000) / 1000.0 * 2 - 1) * daily_random_limits["PV"]
        wind_daily_random_factors[day_value] = 1.0 + (((day_index * 73) % 1000) / 1000.0 * 2 - 1) * daily_random_limits["Wind"]
        locational_marginal_price_daily_random_factors[day_value] = 1.0 + (((day_index * 53) % 1000) / 1000.0 * 2 - 1) * daily_random_limits["LMP"]

    photovoltaic_rows = []
    wind_rows = []
    locational_marginal_price_rows = []

    for timestamp_value in timestamp_index:
        month_value = timestamp_value.month
        hour_value = timestamp_value.hour
        day_key = str(timestamp_value.date())

        photovoltaic_value = (
            annual_means["PV"]
            * seasonal_factors[month_value]["PV"]
            * day_night_factors["PV"][hour_value]
            * photovoltaic_daily_random_factors[day_key]
        )

        wind_value = (
            annual_means["Wind"]
            * seasonal_factors[month_value]["Wind"]
            * day_night_factors["Wind"][hour_value]
            * wind_daily_random_factors[day_key]
        )

        locational_marginal_price_value = (
            annual_means["LMP"]
            * seasonal_factors[month_value]["LMP"]
            * day_night_factors["LMP"][hour_value]
            * locational_marginal_price_daily_random_factors[day_key]
        )

        photovoltaic_value = max(0.0, min(1.0, photovoltaic_value))
        wind_value = max(0.0, min(1.0, wind_value))
        locational_marginal_price_value = max(0.0, locational_marginal_price_value)

        photovoltaic_rows.append({
            "Timestamp": timestamp_value,
            "Year": timestamp_value.year,
            "Month": timestamp_value.month,
            "Day": timestamp_value.day,
            "Hour": timestamp_value.hour,
            "SeasonFactor": seasonal_factors[month_value]["PV"],
            "DayNightFactor": day_night_factors["PV"][hour_value],
            "DailyRandomFactor": photovoltaic_daily_random_factors[day_key],
            "Value": photovoltaic_value,
        })

        wind_rows.append({
            "Timestamp": timestamp_value,
            "Year": timestamp_value.year,
            "Month": timestamp_value.month,
            "Day": timestamp_value.day,
            "Hour": timestamp_value.hour,
            "SeasonFactor": seasonal_factors[month_value]["Wind"],
            "DayNightFactor": day_night_factors["Wind"][hour_value],
            "DailyRandomFactor": wind_daily_random_factors[day_key],
            "Value": wind_value,
        })

        locational_marginal_price_rows.append({
            "Timestamp": timestamp_value,
            "Year": timestamp_value.year,
            "Month": timestamp_value.month,
            "Day": timestamp_value.day,
            "Hour": timestamp_value.hour,
            "SeasonFactor": seasonal_factors[month_value]["LMP"],
            "DayNightFactor": day_night_factors["LMP"][hour_value],
            "DailyRandomFactor": locational_marginal_price_daily_random_factors[day_key],
            "Value": locational_marginal_price_value,
        })

    photovoltaic_dataframe = pd.DataFrame(photovoltaic_rows)
    wind_dataframe = pd.DataFrame(wind_rows)
    locational_marginal_price_dataframe = pd.DataFrame(locational_marginal_price_rows)

    metadata_dataframe = pd.DataFrame([
        {"Parameter": "Year", "Value": year},
        {"Parameter": "NumBuses", "Value": len(bus_profiles_dataframe)},
        {"Parameter": "PV_AnnualMean", "Value": annual_means["PV"]},
        {"Parameter": "Wind_AnnualMean", "Value": annual_means["Wind"]},
        {"Parameter": "LMP_AnnualMean", "Value": annual_means["LMP"]},
        {"Parameter": "PV_DailyRandomLimit", "Value": daily_random_limits["PV"]},
        {"Parameter": "Wind_DailyRandomLimit", "Value": daily_random_limits["Wind"]},
        {"Parameter": "LMP_DailyRandomLimit", "Value": daily_random_limits["LMP"]},
    ])

    with pd.ExcelWriter(profiles_xlsx_path, engine="openpyxl") as writer:
        bus_profiles_dataframe.to_excel(writer, sheet_name="BusProfiles", index=False)
        photovoltaic_dataframe.to_excel(writer, sheet_name="PV", index=False)
        wind_dataframe.to_excel(writer, sheet_name="Wind", index=False)
        locational_marginal_price_dataframe.to_excel(writer, sheet_name="LMP", index=False)
        metadata_dataframe.to_excel(writer, sheet_name="Metadata", index=False)

    log("profiles.xlsx saved successfully")
    log(f"BusProfiles_rows_saved = {len(bus_profiles_dataframe)}")
    log(f"PV_rows_saved = {len(photovoltaic_dataframe)}")
    log(f"Wind_rows_saved = {len(wind_dataframe)}")
    log(f"LMP_rows_saved = {len(locational_marginal_price_dataframe)}")
    log("Returning profiles data")

    profiles_data = {
        "BusProfiles": bus_profiles_dataframe,
        "PV": photovoltaic_dataframe,
        "Wind": wind_dataframe,
        "LMP": locational_marginal_price_dataframe,
        "Metadata": metadata_dataframe,
        "file_path": profiles_xlsx_path,
        "created_new_file": True,
    }

    return profiles_data

def build_network_case(imported_data, profiles, config):
    import os
    import numpy as np
    import pandas as pd
    import datetime

    # Proteção: Garante que o que chegou em `config` é realmente um dicionário
    if not isinstance(config, dict):
        raise TypeError(
            "O argumento 'config' passado para build_network_case deve ser um dicionário. "
            f"Tipo atual recebido: {type(config)}"
        )

    output_dir = os.path.dirname(config["output_xlsx_path"])
    log_path = os.path.join(output_dir, "Network_Data_Imported_debug_log.txt")
    os.makedirs(output_dir, exist_ok=True)

    def write_debug_log(message):
        try:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(message)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] [Build] {message}\n")
        except Exception:
            pass

    def to_dataframe(obj):
        return obj.copy() if isinstance(obj, pd.DataFrame) else pd.DataFrame()

    def normalize_bus_name(value):
        return str(value).strip().split(".")[0] if value is not None else ""

    def to_numeric_series(values, expected_len, fill_value=0.0):
        arr = np.asarray(values, dtype=float).reshape(-1)
        if len(arr) > expected_len:
            return arr[:expected_len]
        return np.pad(arr, (0, max(0, expected_len - len(arr))), constant_values=fill_value)

    def get_profile_series(profile_obj):
        if isinstance(profile_obj, pd.DataFrame):
            if "Value" in profile_obj.columns:
                return profile_obj["Value"].values
            if "LoadScale" in profile_obj.columns:
                return profile_obj["LoadScale"].values
        return profile_obj

    write_debug_log("Iniciando montagem da rede para simulacao...")

    buses_df = to_dataframe(imported_data.get("buses"))
    lines_df = to_dataframe(imported_data.get("lines"))
    transformers_df = to_dataframe(imported_data.get("transformers"))
    loads_df = to_dataframe(imported_data.get("loads"))
    pvsystems_df = to_dataframe(imported_data.get("pvsystems"))
    generators_df = to_dataframe(imported_data.get("generators"))

    if "BusName" in buses_df.columns:
        buses_df["BusName"] = buses_df["BusName"].astype(str).map(normalize_bus_name)
    if "Bus1" in lines_df.columns:
        lines_df["Bus1"] = lines_df["Bus1"].astype(str).map(normalize_bus_name)
    if "Bus2" in lines_df.columns:
        lines_df["Bus2"] = lines_df["Bus2"].astype(str).map(normalize_bus_name)
    if "Bus1" in transformers_df.columns:
        transformers_df["Bus1"] = transformers_df["Bus1"].astype(str).map(normalize_bus_name)
    if "Bus2" in transformers_df.columns:
        transformers_df["Bus2"] = transformers_df["Bus2"].astype(str).map(normalize_bus_name)
    if "Bus" in loads_df.columns:
        loads_df["Bus"] = loads_df["Bus"].astype(str).map(normalize_bus_name)

    prof_cfg = config["profiles"]

    temporal_keys = (
        prof_cfg["solar_key_candidates"]
        + prof_cfg["wind_key_candidates"]
        + prof_cfg["price_key_candidates"]
    )
    temporal_key = next((k for k in temporal_keys if k in profiles), None)
    horizon = len(get_profile_series(profiles[temporal_key])) if temporal_key else 8760

    load_key = next((k for k in prof_cfg["load_key_candidates"] if k in profiles and k.lower() != "busprofiles"), None)
    solar_key = next((k for k in prof_cfg["solar_key_candidates"] if k in profiles), None)
    wind_key = next((k for k in prof_cfg["wind_key_candidates"] if k in profiles), None)
    price_key = next((k for k in prof_cfg["price_key_candidates"] if k in profiles), None)

    load_profile = to_numeric_series(get_profile_series(profiles[load_key]), horizon, 1.0) if load_key else np.ones(horizon)
    solar_profile = to_numeric_series(get_profile_series(profiles[solar_key]), horizon, 0.0) if solar_key else np.zeros(horizon)
    wind_profile = to_numeric_series(get_profile_series(profiles[wind_key]), horizon, 0.0) if wind_key else np.zeros(horizon)
    price_profile = to_numeric_series(get_profile_series(profiles[price_key]), horizon, np.nan) if price_key else np.full(horizon, np.nan)

    dt = float(config["profiles"]["time_step_hours"])
    default_freq = float(config.get("default_frequency_hz", 60.0))

    for col in ["KW", "Kvar"]:
        if col in loads_df.columns:
            loads_df[col] = pd.to_numeric(loads_df[col], errors="coerce")

    if not loads_df.empty and "Bus" in loads_df.columns:
        bus_loads = loads_df.groupby("Bus", dropna=False)[["KW", "Kvar"]].sum(min_count=1).reset_index()
        bus_loads.rename(columns={"KW": "BaseLoadKW", "Kvar": "BaseLoadKvar"}, inplace=True)
        buses_enriched_df = buses_df.merge(
            bus_loads,
            how="left",
            left_on="BusName",
            right_on="Bus"
        ).drop(columns=["Bus"], errors="ignore")
    else:
        buses_enriched_df = buses_df.copy()
        buses_enriched_df["BaseLoadKW"] = 0.0
        buses_enriched_df["BaseLoadKvar"] = 0.0

    buses_enriched_df["BaseLoadKW"] = buses_enriched_df.get("BaseLoadKW", pd.Series([0.0]*len(buses_enriched_df))).fillna(0.0)
    buses_enriched_df["BaseLoadKvar"] = buses_enriched_df.get("BaseLoadKvar", pd.Series([0.0]*len(buses_enriched_df))).fillna(0.0)

    total_base_load_kw = float(buses_enriched_df["BaseLoadKW"].sum())
    total_base_load_kvar = float(buses_enriched_df["BaseLoadKvar"].sum())

    existing_pv_kw = float(pvsystems_df["Pmpp"].sum()) if "Pmpp" in pvsystems_df.columns and not pvsystems_df.empty else 0.0
    existing_wind_kw = float(generators_df["KW"].sum()) if "KW" in generators_df.columns and not generators_df.empty else 0.0

    max_der_penetration = config["optimization"]["der_penetration_limit_percent"] / 100.0
    total_load_kw_series = total_base_load_kw * load_profile
    peak_load = float(np.max(total_load_kw_series)) if total_base_load_kw > 0 else 0.0
    max_total_der_allowed_kw = peak_load * max_der_penetration

    buses_enriched_df["IsLowVoltage"] = buses_enriched_df["BaseKV"].apply(
        lambda kv: True if pd.notna(kv) and float(kv) < 1.0 else False
    ) if "BaseKV" in buses_enriched_df.columns else False
    
    lv_buses = buses_enriched_df[buses_enriched_df["IsLowVoltage"]].copy()

    buses_enriched_df["Installed_PV_KW"] = 0.0
    buses_enriched_df["Installed_Wind_KW"] = 0.0
    buses_enriched_df["Installed_BESS_kWh"] = 0.0

    new_pv_kw = 0.0
    new_wind_kw = 0.0
    bess_total_kwh = 0.0

    if config["optimization"]["allow_generation_creation"] and not lv_buses.empty:
        remaining_capacity = max(0.0, max_total_der_allowed_kw - existing_pv_kw - existing_wind_kw)

        pv_capex = config["economic_parameters"]["solar_pv"]["capex_usd_kw"]
        wind_capex = config["economic_parameters"]["wind"]["capex_usd_kw"]

        if wind_capex < pv_capex:
            wind_per_bus = min(
                config["economic_parameters"]["wind"]["max_install_kw_per_bus"],
                remaining_capacity / len(lv_buses) if len(lv_buses) > 0 else 0.0
            )
            buses_enriched_df.loc[buses_enriched_df["IsLowVoltage"], "Installed_Wind_KW"] = wind_per_bus
            new_wind_kw = wind_per_bus * len(lv_buses)
            remaining_capacity -= new_wind_kw

            pv_per_bus = min(
                config["economic_parameters"]["solar_pv"]["max_install_kw_per_bus"],
                remaining_capacity / len(lv_buses) if len(lv_buses) > 0 else 0.0
            )
            buses_enriched_df.loc[buses_enriched_df["IsLowVoltage"], "Installed_PV_KW"] = pv_per_bus
            new_pv_kw = pv_per_bus * len(lv_buses)
        else:
            pv_per_bus = min(
                config["economic_parameters"]["solar_pv"]["max_install_kw_per_bus"],
                remaining_capacity / len(lv_buses) if len(lv_buses) > 0 else 0.0
            )
            buses_enriched_df.loc[buses_enriched_df["IsLowVoltage"], "Installed_PV_KW"] = pv_per_bus
            new_pv_kw = pv_per_bus * len(lv_buses)
            remaining_capacity -= new_pv_kw

            wind_per_bus = min(
                config["economic_parameters"]["wind"]["max_install_kw_per_bus"],
                remaining_capacity / len(lv_buses) if len(lv_buses) > 0 else 0.0
            )
            buses_enriched_df.loc[buses_enriched_df["IsLowVoltage"], "Installed_Wind_KW"] = wind_per_bus
            new_wind_kw = wind_per_bus * len(lv_buses)

    if config["optimization"]["allow_storage"] and not lv_buses.empty:
        bess_per_bus = config["economic_parameters"]["battery_storage"]["max_install_kwh_per_bus"]
        buses_enriched_df.loc[buses_enriched_df["IsLowVoltage"], "Installed_BESS_kWh"] = bess_per_bus
        bess_total_kwh = bess_per_bus * len(lv_buses)

    network_case = {
        "config": config,
        "profiles": profiles,
        "horizon": horizon,
        "dt_hours": dt,
        "default_frequency_hz": default_freq,
        "buses_df": buses_df,
        "lines_df": lines_df,
        "transformers_df": transformers_df,
        "loads_df": loads_df,
        "pvsystems_df": pvsystems_df,
        "generators_df": generators_df,
        "buses_exact_df": buses_enriched_df,
        "load_profile": load_profile,
        "solar_profile": solar_profile,
        "wind_profile": wind_profile,
        "price_profile": price_profile,
        "total_base_load_kw": total_base_load_kw,
        "total_base_load_kvar": total_base_load_kvar,
        "existing_pv_kw": existing_pv_kw,
        "existing_wind_kw": existing_wind_kw,
        "new_pv_kw": new_pv_kw,
        "new_wind_kw": new_wind_kw,
        "bess_total_kwh": bess_total_kwh,
        "peak_load_kw": peak_load,
        "output_xlsx_path": config["output_xlsx_path"],
        "log_path": log_path
    }

    write_debug_log("Montagem da rede concluida com sucesso.")
    return network_case

def run_stochastic_simulation(network_case):
    import os
    import numpy as np
    import pandas as pd
    import datetime

    config = network_case["config"]
    output_dir = os.path.dirname(config["output_xlsx_path"])
    log_path = network_case.get("log_path", os.path.join(output_dir, "Network_Data_Imported_debug_log.txt"))
    os.makedirs(output_dir, exist_ok=True)

    def write_debug_log(message):
        try:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(message)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] [Sim] {message}\n")
        except Exception:
            pass

    write_debug_log("Iniciada simulacao estocastica com rede pre-montada...")

    buses_enriched_df = network_case["buses_exact_df"].copy()
    lines_df = network_case["lines_df"].copy()
    transformers_df = network_case["transformers_df"].copy()

    horizon = int(network_case["horizon"])
    dt = float(network_case["dt_hours"])
    default_freq = float(network_case["default_frequency_hz"])

    load_profile = np.asarray(network_case["load_profile"], dtype=float)
    solar_profile = np.asarray(network_case["solar_profile"], dtype=float)
    wind_profile = np.asarray(network_case["wind_profile"], dtype=float)
    price_profile = np.asarray(network_case["price_profile"], dtype=float)

    total_base_load_kw = float(network_case["total_base_load_kw"])
    total_base_load_kvar = float(network_case["total_base_load_kvar"])
    peak_load = float(network_case["peak_load_kw"])

    system_pv_kw = float(network_case["existing_pv_kw"] + network_case["new_pv_kw"])
    system_wind_kw = float(network_case["existing_wind_kw"] + network_case["new_wind_kw"])
    bess_total_kwh = float(network_case["bess_total_kwh"])

    total_load_kw_series = total_base_load_kw * load_profile
    pv_generation_series = system_pv_kw * solar_profile
    wind_generation_series = system_wind_kw * wind_profile
    total_generation_series = pv_generation_series + wind_generation_series
    base_net_load = total_load_kw_series - total_generation_series

    write_debug_log("Executando arbitragem/peak shaving do BESS...")

    battery_eff = config["economic_parameters"]["battery_storage"]["efficiency_round_trip"]
    allow_storage = config["optimization"]["allow_storage"]
    allow_curtailment = config["optimization"]["allow_curtailment"]

    battery_soc_max = bess_total_kwh * config["economic_parameters"]["battery_storage"]["soc_max"]
    battery_soc_min = bess_total_kwh * config["economic_parameters"]["battery_storage"]["soc_min"]
    current_soc = battery_soc_min

    battery_dispatch_kw = np.zeros(horizon)
    soc_series = np.zeros(horizon)
    grid_import_kw = np.zeros(horizon)
    curtailed_kw = np.zeros(horizon)

    valid_prices = not np.all(np.isnan(price_profile))
    if valid_prices:
        p_charge_limit = np.nanpercentile(price_profile, 20)
        p_discharge_limit = np.nanpercentile(price_profile, 80)
    else:
        p_charge_limit = np.nanpercentile(base_net_load, 20)
        p_discharge_limit = np.nanpercentile(base_net_load, 80)

    bess_max_kw = bess_total_kwh / 4.0 if bess_total_kwh > 0 else 0.0

    for t in range(horizon):
        net_load = base_net_load[t]
        control_signal = price_profile[t] if valid_prices else net_load
        bess_kw = 0.0

        if control_signal >= p_discharge_limit and allow_storage and current_soc > battery_soc_min:
            available_kwh = current_soc - battery_soc_min
            bess_kw = min(bess_max_kw, available_kwh / dt)

            if allow_curtailment and bess_kw > net_load and net_load > 0:
                bess_kw = net_load

        elif control_signal <= p_charge_limit and allow_storage and current_soc < battery_soc_max:
            space_kwh = battery_soc_max - current_soc
            bess_kw = -min(bess_max_kw, space_kwh / (dt * battery_eff))

        if net_load < 0 and allow_storage and current_soc < battery_soc_max:
            space_kwh = battery_soc_max - current_soc
            current_charge_kw = abs(min(0.0, bess_kw))
            additional_charge_kw = min(abs(net_load), space_kwh / (dt * battery_eff) - current_charge_kw)
            if additional_charge_kw > 0:
                bess_kw = -max(current_charge_kw, additional_charge_kw)

        if bess_kw >= 0:
            current_soc -= bess_kw * dt
        else:
            current_soc += abs(bess_kw) * battery_eff * dt

        current_soc = min(max(current_soc, battery_soc_min), battery_soc_max) if bess_total_kwh > 0 else 0.0

        final_import = net_load - bess_kw

        if final_import < 0 and allow_curtailment:
            curtailed_kw[t] = abs(final_import)
            final_import = 0.0

        grid_import_kw[t] = final_import
        battery_dispatch_kw[t] = bess_kw
        soc_series[t] = current_soc

    write_debug_log("Montando dataframes de saida...")

    system_time_df = pd.DataFrame({
        "TimeStep": np.arange(horizon, dtype=int),
        "TotalLoadKW": total_load_kw_series,
        "PVGenerationKW": pv_generation_series,
        "WindGenerationKW": wind_generation_series,
        "BatteryDispatchKW": battery_dispatch_kw,
        "BatterySOC_kWh": soc_series,
        "GridImportKW": grid_import_kw,
        "CurtailedKW": curtailed_kw
    })

    bus_names = buses_enriched_df["BusName"].values
    p_shares = (
        buses_enriched_df["BaseLoadKW"].values / total_base_load_kw
        if total_base_load_kw > 0 else np.zeros(len(bus_names))
    )
    q_shares = (
        buses_enriched_df["BaseLoadKvar"].values / total_base_load_kvar
        if total_base_load_kvar > 0 else np.zeros(len(bus_names))
    )

    p_kw = np.outer(p_shares, grid_import_kw.ravel())
    q_kvar = np.outer(q_shares, (total_base_load_kvar * load_profile).ravel())
    s_kva = np.sqrt(p_kw**2 + q_kvar**2)
    pf_series = np.divide(p_kw, s_kva, out=np.ones_like(p_kw), where=s_kva != 0)

    safe_peaks = np.maximum(1e-6, peak_load * p_shares)
    # Reformata safe_peaks para (436, 1) para fazer a divisao espalhando pelas 8760 colunas
    safe_peaks_2d = safe_peaks.reshape(-1, 1)
    v_pu = 1.0 - 0.05 * (p_kw / safe_peaks_2d)
    v_pu = np.clip(v_pu, config["voltage_limits_pu"]["min"], config["voltage_limits_pu"]["max"])

    bus_time_df = pd.DataFrame({
        "TimeStep": np.tile(np.arange(horizon), len(bus_names)),
        "BusName": np.repeat(bus_names, horizon),
        "Vpu": v_pu.ravel(),
        "PkW": p_kw.ravel(),
        "Qkvar": q_kvar.ravel(),
        "SkVA": s_kva.ravel(),
        "PowerFactor": pf_series.ravel(),
        "FrequencyHz": default_freq
    })

    line_names = lines_df["Name"].values if not lines_df.empty and "Name" in lines_df.columns else np.array([f"Line_{i}" for i in range(len(lines_df))])
    norm_amps = lines_df["NormAmps"].fillna(1000.0).values if not lines_df.empty and "NormAmps" in lines_df.columns else np.full(len(lines_df), 1000.0)
    r1_vals = lines_df["R1"].fillna(0.01).values if not lines_df.empty and "R1" in lines_df.columns else np.full(len(lines_df), 0.01)

    base_current_index = grid_import_kw / max(1e-6, peak_load)
    
    if len(line_names) > 0:
        i_amps = np.outer(norm_amps * 0.4, base_current_index.ravel())
        limit_amps_arr = np.repeat(norm_amps, horizon)
        
        lines_time_df = pd.DataFrame({
            "TimeStep": np.tile(np.arange(horizon), len(line_names)),
            "LineName": np.repeat(line_names, horizon),
            "CurrentAmps": i_amps.ravel(),
            "LimitAmps": limit_amps_arr,
            "LoadingPercent": (i_amps.ravel() / limit_amps_arr) * 100,
            "PLosskW": (i_amps**2 * np.repeat(r1_vals, horizon).reshape(len(r1_vals), horizon) * 0.001).ravel(),
            "Overloaded": i_amps.ravel() > limit_amps_arr
        })
    else:
        lines_time_df = pd.DataFrame()

    transf_names = transformers_df["Name"].values if not transformers_df.empty and "Name" in transformers_df.columns else np.array([f"Trafo_{i}" for i in range(len(transformers_df))])
    kva_caps = transformers_df["kVA"].fillna(1000.0).values if not transformers_df.empty and "kVA" in transformers_df.columns else np.full(len(transformers_df), 1000.0)

    if len(transf_names) > 0:
        s_flow = np.outer(kva_caps * 0.6, base_current_index.ravel())
        limit_kva_arr = np.repeat(kva_caps, horizon)

        transf_time_df = pd.DataFrame({
            "TimeStep": np.tile(np.arange(horizon), len(transf_names)),
            "TransfName": np.repeat(transf_names, horizon),
            "SFlowkVA": s_flow.ravel(),
            "LimitkVA": limit_kva_arr,
            "LoadingPercent": (s_flow.ravel() / limit_kva_arr) * 100,
            "Overloaded": s_flow.ravel() > limit_kva_arr
        })
    else:
        transf_time_df = pd.DataFrame()

    der_locations_df = buses_enriched_df[
        ["BusName", "IsLowVoltage", "Installed_PV_KW", "Installed_Wind_KW", "Installed_BESS_kWh"]
    ].copy() if not buses_enriched_df.empty else pd.DataFrame()

    if not der_locations_df.empty:
        der_locations_df = der_locations_df[
            (der_locations_df["Installed_PV_KW"] > 0) |
            (der_locations_df["Installed_Wind_KW"] > 0) |
            (der_locations_df["Installed_BESS_kWh"] > 0)
        ]

    scalar_results_df = pd.DataFrame([
        {"Metric": "HorizonSteps", "Value": horizon},
        {"Metric": "PeakTotalLoadKW", "Value": peak_load},
        {"Metric": "InstalledPVKW", "Value": system_pv_kw},
        {"Metric": "InstalledWindKW", "Value": system_wind_kw},
        {"Metric": "InstalledBatterykWh", "Value": bess_total_kwh},
        {"Metric": "TotalBaseLoadKW", "Value": total_base_load_kw},
        {"Metric": "TotalBaseLoadKvar", "Value": total_base_load_kvar},
        {"Metric": "TotalCurtailedEnergykWh", "Value": float(np.sum(curtailed_kw) * dt)},
        {"Metric": "TotalGridImportEnergykWh", "Value": float(np.sum(grid_import_kw) * dt)}
    ])

    write_debug_log("Salvando arquivos auxiliares...")
    bus_time_df.to_csv(os.path.join(output_dir, "BusTimeSeries.csv"), index=False)
    if not lines_time_df.empty:
        lines_time_df.to_csv(os.path.join(output_dir, "LinesTimeSeries.csv"), index=False)
    if not transf_time_df.empty:
        transf_time_df.to_csv(os.path.join(output_dir, "TransfTimeSeries.csv"), index=False)

    write_debug_log("Salvando arquivo Excel principal...")
    try:
        with pd.ExcelWriter(config["output_xlsx_path"], engine="openpyxl") as writer:
            scalar_results_df.to_excel(writer, sheet_name="ScalarResults", index=False)
            system_time_df.to_excel(writer, sheet_name="SystemTimeSeries", index=False)
            
            if not der_locations_df.empty:
                der_locations_df.to_excel(writer, sheet_name="DERLocations", index=False)
                
            if not buses_enriched_df.empty:
                buses_enriched_df.to_excel(writer, sheet_name="BusesExact", index=False)
        write_debug_log("Simulacao finalizada com exito.")
    except PermissionError:
        write_debug_log("ERRO DE PERMISSAO: o arquivo Excel de destino estava aberto durante a geracao.")

    optimization_output = {
        "config_used": config,
        "scalar_results": scalar_results_df,
        "system_time_series": system_time_df,
        "bus_time_series": bus_time_df,
        "lines_time_series": lines_time_df,
        "transf_time_series": transf_time_df,
        "der_locations": der_locations_df,
        "buses_exact": buses_enriched_df,
        "output_xlsx_path": config["output_xlsx_path"]
    }

    return optimization_output


if __name__ == "__main__":
    imported_data = load_network_from_master()
    profiles = load_or_create_profiles(imported_data)
    network_case = build_network_case(imported_data, profiles, CONFIG)
    optimization_output = run_stochastic_simulation(network_case)