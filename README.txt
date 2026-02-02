===========================================================
Stochastic Optimization Framework for VDER Integration
===========================================================

1. OVERVIEW
-----------
This project provides an integrated simulation framework to 
optimize and analyze Distributed Energy Resources (DERs). 
The system couples a MILP stochastic optimizer with an 
OpenDSS digital twin to evaluate Technical, Economic, and 
Social Equity impacts.

2. SYSTEM REQUIREMENTS
----------------------
A. Core Engines:
   * OpenDSS: Must be accessible via 'py_dss_interface'.
   * CPLEX: Required solver for 'docplex'.

B. Directory Structure:
   Ensure the following hierarchy exists:
   [Project Root]
     ├── run.bat
     ├── requirements.txt
     ├── code/
     └── Iowa_Distribution_Test_Systems/

3. INSTALLATION & EXECUTION INSTRUCTIONS (STRICT)
-------------------------------------------------
Follow these steps strictly to configure the environment 
and run the simulation.

STEP 1: PYTHON 3.10 INSTALLATION
--------------------------------
1. Download the Python 3.10 installer.
2. Run the installer and select "Customize installation".
3. CRITICAL: Change the installation path to exactly:
   
   C:\Python31

4. Complete the installation.

STEP 2: VIRTUAL ENVIRONMENT SETUP
---------------------------------
1. Open a Command Prompt (terminal) inside this [Project Root] folder.
   (Tip: You can type 'cmd' in the folder address bar).
2. Execute the following command to create the virtual environment:

   C:\Python31\python.exe -m venv venv310

   *Note: This will create a folder named 'venv310' in the project root.*

STEP 3: LAUNCHING THE SYSTEM
----------------------------
1. In the Project Root, double-click the file: 'run.bat'.
2. The script will verify that 'venv310' exists.
3. When prompted:
   "Do you want to verify/install libraries from requirements.txt? (Y/N)"
   
   Type: Y
   (Press Enter)

4. The script will automatically install all dependencies inside 
   the virtual environment and launch the dashboard.

STEP 4: ACCESS
--------------
The Streamlit interface will open automatically in your 
default web browser (http://localhost:8501).

4. TECHNICAL FEATURES
---------------------
* Stochastic Scenarios: Generates paths for load and renewable 
  uncertainty using Gaussian noise.
* Dynamic Network Import: Automatically converts 'Master.dss' 
  topology into an Excel database on startup.
* Topological Zoning: Classifies grid nodes as Urban, Mixed, 
  or Rural using K-Means clustering.
* Smart Batch Mode: Checks 'MASTER_Hourly_Results.csv' logs 
  to skip previously completed weight combinations.

===========================================================
End of Documentation
===========================================================