===========================================================
Stochastic Optimization Framework for VDER Integration
===========================================================

## ⚠️ MANUAL PREREQUISITES (One-time)

❌ CPLEX → IBM website
- Copy the Cplex license files directly to the cplex_lib folder. These files are usually located in the folder C:\Program Files\IBM\ILOG\CPLEX_Studio2211\cplex\python\3.10\x64_win64
❌ OpenDSS → DLLs in Windows PATH
- Paste the OpenDSS file into C:\Program Files\OpenDSS

## 🚀 QUICK START

**STEP 1:** Double-click `run.bat`  

**STEP 2:** Answer prompts:  
Install Portable Environment? (Y/N): → Y (first time)  
Check requirements.txt? (Y/N): → Y

**STEP 3:** Firefox opens automatically:  
Dashboard: http://localhost:8501 ✅  
Visualizer: http://localhost:8502 ✅  

## 📁 WHAT run.bat DOES AUTOMATICALLY  

✅ Downloads Python 3.10 Portable  
✅ Creates venv310 environment  
✅ Installs pip + wheel + libraries  
✅ Downloads Firefox Portable browser  
✅ Starts Dashboard + Visualizer  
✅ Opens both apps in browser  

## 🔧 TROUBLESHOOTING  

BLANK PAGE? → FirefoxPortable.exe opens automatically (IE incompatible)  
CPLEX ERROR? → Install IBM CPLEX Community Edition  
OpenDSS ERROR? → Add OpenDSS DLLs to PATH  
"Port in use"? → Close other Streamlit apps (Ctrl+C)  

## 🛠️ TECHNICAL FEATURES  

- Stochastic load/renewable uncertainty  
- OpenDSS→Excel topology conversion  
- K-Means grid zoning (Urban/Mixed/Rural)  
- Smart batch mode (resume capability)  

===========================================================  
**🔥 Fully Portable: Python + Firefox + Libs = AUTO**  
**📦 Manual: Only CPLEX + OpenDSS DLLs**  