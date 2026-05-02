import py_dss_interface

dss = py_dss_interface.DSS()

print("py_dss_interface version check:")
try:
    import py_dss_interface
    print(py_dss_interface.__version__)
except Exception:
    print("version not available")

print("\nCircuit:")
print(dir(dss.circuit))

print("\nLines:")
print(dir(dss.lines))

print("\nLoads:")
print(dir(dss.loads))

print("\nTransformers:")
print(dir(dss.transformers))

print("\nBus:")
print(dir(dss.bus))

print("\nCktElement:")
print(dir(dss.cktelement))

print("\nRegControls:")
print(dir(dss.regcontrols))

print("\nLineCodes:")
print(dir(dss.linecodes))

print("\nPVSystems:")
print(dir(dss.pvsystems))

print("\nVSources:")
print(dir(dss.vsources))