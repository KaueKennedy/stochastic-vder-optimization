import os
import re
import math
import shutil
import tempfile
import datetime
import pandas as pd
import py_dss_interface


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

if __name__ == "__main__":
    imported_data = load_network_from_master()