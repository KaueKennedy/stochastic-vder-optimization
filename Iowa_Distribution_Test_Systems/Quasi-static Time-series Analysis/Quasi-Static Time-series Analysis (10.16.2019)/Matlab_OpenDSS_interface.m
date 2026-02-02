% Note: This file is a Matlab-OpenDSS interface file used to solve the quasi-static time-series power flow for the 240-bus distribution test system. For the correponding OpenDSS model, please see "OpenDSS Model.zip" file.
%       The active powers and reactive powers are calculated from the real hourly smart meter measurements, and the data ranges from January 2017 to December 2017 (8760 hours). 
% Specifically,
% 1. The nodal active powers and reative powers corresponding to the three feeders (Feeder A, Feeder B, and Feeder C) are stored in FeederA_P.mat, FeederA_Q.mat, FeederB_P.mat, FeederB_Q.mat, FeederC_P.mat and FeederC_Q.mat. 
%    In addition, FeederA_P_Q_Header.mat, FeederB_P_Q_Header.mat and FeederC_P_Q_Header.mat specify the bus names correspond to the columns in the six active power and reactive power .mat files.
% 2. The power flow result obtained via Matlab-OpenDSS interface includes: 
%    bus voltages, element losses, total power of the entire system, line currents, line powers, tap changer positions.




clear;

%% Read load data
load FeederA_P;     % Active powers correpsonding to Feeder A's buses  
load FeederA_Q;     % Reactive powers correpsonding to Feeder A's buses
load FeederB_P;     % Active powers correpsonding to Feeder B's buses
load FeederB_Q;     % Reactive powers correpsonding to Feeder B's buses
load FeederC_P;     % Active powers correpsonding to Feeder C's buses
load FeederC_Q;     % Reactive powers correpsonding to Feeder C's buses

%% Build the Matlab-OpenDSS COM interface
DSSObj=actxserver('OpenDSSEngine.DSS');            % Register the COM server (initialization)
if ~DSSObj.Start(0)                                % Start the OpenDSS, and if the registration is unsuccessful, stop the program and remind the user
    disp('Unable to start OpenDSS Engine');
    return
end
DSSText = DSSObj.Text;                             % Define a text interface variable
DSSCircuit = DSSObj.ActiveCircuit;                 % Define a circuit interface variable
DSSText.Command='Compile "C:\Users\fbu\Desktop\Project\Test system 1\Matlab_OpenDSS_Interface\1015\Master.dss"';   % Specify the directory of OpenDSS master file 

%% Define variables to collect the power flow results
bus_voltages_rect = [];                   % Bus voltage in rectangular coordinate
bus_voltage_magni_pu = [];                % Bus voltage magnitude in per unit.
line_currents = {};                       % Line current
line_powers = {};                         % Line power
elem_names = [];                          % Element names
elem_losses = [];                         % Element loss
total_power = [];                         % System total power                
i_notconverged = 0;                       % Define a variable to record the number of unconverged snapshot power flow solutions
Tap_position_collect = [];                % Tap changer position


%% Specify load buses (buses with load)
FeederA_bus_with_load = 1003:1017;                                                                                                                     % Buses of Feeder A that have loads
FeederB_bus_with_load = [2002:2003, 2005, 2008:2011, 2014:2018, 2020, 2022:2025, 2028:2032, 2034:2035, 2037, 2040:2043, 2045:2056, 2058:2060];         % Buses of Feeder B that have loads
FeederC_bus_with_load = [3002, 3004, 3006:3007, 3009:3014, 3016:3021, 3023:3029, 3031:3039, 3041:3045, 3047:3052, 3054, 3056:3067, 3070:3074, 3077:3078, 3081, ...
                         3083:3091, 3093:3099, 3101:3106, 3108:3112, 3114:3117, 3120:3132, 3134:3138, 3141:3155, 3157:3162];                           % Buses of Feeder C that have loads


%% Solve quasi-static time-series power flow via Matlab-OpenDSS interface and collect results
n = length(FeederA_P(:, 1));                        % Number of hours in one year, i.e., 8760
for i = 1:n
  
      %% For each load of Feeder A, set kW and kVar 
      for k = 1:length(FeederA_bus_with_load)           % From the 1st bus with load to the last bus with load
          bus_num = FeederA_bus_with_load(1,k);         % Bus No.
          DSSText.command=[[char('load.Load_'), num2str(bus_num), char('.kW=')]  num2str(FeederA_P(i, bus_num-1000)) ' kvar='  num2str(FeederA_Q(i, bus_num-1000)) ''];  % Build bus name and set corresponding kW and kVar
                                                                                                                                                   %  bus_num-1000 specifies the column number that the power corresponds to 
      end    
    
    
      %% For each load of Feeder B, set kW and kVar 
      for k = 1:length(FeederB_bus_with_load)          % From the 1st bus with load to the last bus with load
          bus_num = FeederB_bus_with_load(1,k);        % Bus No.
          DSSText.command=[[char('load.Load_'), num2str(bus_num), char('.kW=')]  num2str(FeederB_P(i, bus_num-2000)) ' kvar='  num2str(FeederB_Q(i, bus_num-2000)) ''];   % Build bus name and set corresponding kW and kVar
                                                                                                                                                   %  bus_num-2000 specifies the column number that the power corresponds to
      end     
    
    
      %% For each load of Feeder C, set kW and kVar 
      for k = 1:length(FeederC_bus_with_load)          % From the 1st bus with load to the last bus with load
          bus_num = FeederC_bus_with_load(1,k);        % Bus No.
          DSSText.command=[[char('load.Load_'), num2str(bus_num), char('.kW=')]  num2str(FeederC_P(i, bus_num-3000)) ' kvar='  num2str(FeederC_Q(i, bus_num-3000)) ''];   % Build bus name and set corresponding kW and kVar
                                                                                                                                                   %  bus_num-3000 specifies the column number that the power corresponds to
      end


      %% Solve snapshot power flow
      DSSText.Command='solve';


      %% Convergence checking and record the number of uncoverged snapshot power flow
      DSSSolution = DSSCircuit.Solution;              % Obtain solution information from the interface
      if ~DSSSolution.Converged                       % Check whether the power flow computation converges
          fprintf('The Circuit did not converge. \n');
          i_notconverged = i_notconverged + 1;        % calculate the total number of unconverged power flow computations
          continue;
      end
      
      
      %% Collect power flow results 
      % Collect bus names 
      bus_names = DSSCircuit.AllNodenames;   
      
      % Collect all bus voltages in rectangular coordinate
      bus_voltage_temp = DSSCircuit.AllBusVolts;                                                  % Obtain rectangular bus voltage in each snapshot power flow solution            
      bus_voltages_rect = [bus_voltages_rect; bus_voltage_temp];                                  % Collect rectangular bus voltages in all snapshot power flow solutions 

      % Collect all bus voltage magnitude in p.u.
      bus_voltage_magni_pu_temp = DSSCircuit.AllBusVmagPu;                                        % Obtain bus voltage in p.u. in each snapshot power flow solution 
      bus_voltage_magni_pu = [bus_voltage_magni_pu; bus_voltage_magni_pu_temp];                   % Collect bus voltages in p.u. in all snapshot power flow solutions
      
      % Collect element names and losses
      elem_names = DSSCircuit.AllElementNames;                                                    % Obtain element names
      elem_loss_temp = DSSCircuit.AllElementLosses;                                               % Obtain element losses in each snapshot power flow solution
      elem_losses = [elem_losses; elem_loss_temp];                                                % Collect element losses in all snapshot power flow solutions

      % Collect total power of the entire system
      total_power_temp = DSSCircuit.TotalPower;                                                   % Obtain total power of the entire system in each snapshot power flow solution
      total_power = [total_power; total_power_temp];                                              % Collect total power of the entire system in all snapshot power flow solutions

      % Collect currents and powers of all lines
       currents_DSS_Lines = [];                                                                   % Define a variable to collect line currents in each snapshot power flow solution
       powers_DSS_Lines = [];                                                                     % Define a variable to collect line powers in each snapshot power flow solution
       DSSLines = DSSObj.ActiveCircuit.Lines;                                                     % Specify that the currently activated objects are lines
       DSSActiveCktElement = DSSObj.ActiveCircuit.ActiveCktElement;                               % Returns an interface to the active circuit element (lines).

       
       line_names = {};                              % Names of lines
       line_I_points_nums = [];                      % For each line, define a variable to collect the number of current (or power) variables in rectangular coordinate, e.g., a single phase line has 4 current variables,... 
                                                     % ... i.e., real and image parts of the current which flows into the head of the line, real and image parts of the current which flows out the end of the line 
       i_Line = DSSLines.First;                      % Initializing line NO. as the first line
       while i_Line > 0                              % From the 1st line to the last line
            currents_DSS_Lines = [currents_DSS_Lines, DSSActiveCktElement.Currents];                % Collect line currents in each snapshot power flow solution
            powers_DSS_Lines= [powers_DSS_Lines, DSSActiveCktElement.Powers];                       % Collect line powers in each snapshot power flow solution
            line_names{i_Line, 1} = DSSActiveCktElement.NAME;                                       % Collect line names in each snapshot power flow solution
            line_I_points_nums = [line_I_points_nums; length(DSSActiveCktElement.Currents)];        % Collect the total number of variables correponsing to each line in each snapshot power flow solution
            i_Line = DSSLines.Next;                                                                 % Move to next line in each snapshot power flow solution
       end

       
       line_curr_temp = {};       
       line_power_temp = {};
       num_lines = length(line_names);
       for j = 1:num_lines                                                          % find starting and ending indices of currents for each line 
           indx_str = 1 + sum(line_I_points_nums(1:j-1));
           indx_end = sum(line_I_points_nums(1:j));
           line_curr_temp{1, j} = currents_DSS_Lines(indx_str:indx_end);
           line_power_temp{1, j} = powers_DSS_Lines(indx_str:indx_end);
       end
       
       
      line_currents(i,:) = line_curr_temp;                                          % Collect line currents in all snapshot power flow solutions
      line_powers(i,:) = line_power_temp;                                           % Collect line powers in all snapshot power flow solutions      
       
      
      
      % Collect tap positions
      DSSCircuit.RegControls.Name = 'Reg_contr_A';                          % Specify the name of tap changer contoller from which we want to get tap postion                         
      TapChanger1_temp = DSSCircuit.RegControls.TapNumber;                  % Obtain tap position of tap changer (Phase A)
      DSSCircuit.RegControls.Name = 'Reg_contr_B';                          % Specify the name of tap changer contoller from which we want to get tap postion
      TapChanger2_temp = DSSCircuit.RegControls.TapNumber;                  % Obtain tap position of tap changer (Phase B)
      DSSCircuit.RegControls.Name = 'Reg_contr_C';                          % Specify the name of tap changer contoller from which we want to get tap postion
      TapChanger3_temp = DSSCircuit.RegControls.TapNumber;                  % Obtain tap position of tap changer (Phase C)
      Tap_position_collect = [Tap_position_collect; [TapChanger1_temp, TapChanger2_temp, TapChanger3_temp]];   % Collect tap changers positions in all snapshot power flow solutions
      

end

fprintf('The number of snapshot power flow soultions that do not converge is: %d. \n', i_notconverged);                 % Print the total number of unconverged power flow solutions
