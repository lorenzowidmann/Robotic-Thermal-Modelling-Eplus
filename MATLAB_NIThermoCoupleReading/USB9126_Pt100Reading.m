clear; clc; close all;

%% --------------------------- VARIABLES ------------------------------- %%
% save data name
saveName = "Acq01";

%% --------------------------- ACQUISITION ----------------------------- %%
% create data acquisition object for specified vendor (in this case "National Instrument")
dq = daq("ni");

% set the acquisition rate
dq.Rate = 2;    % [Hz]

% define the input channel
ch0 = dq.addinput("Dev2", "ai0", "RTD");
ch0.R0 = 100;
ch0.RTDType = "Pt3851";
ch0.RTDConfiguration = "FourWire";

% create the data vector
data_vect = [0];

% acquire data
fprintf("-------------------------------\n");
fprintf("Start acquisition\n");

fg = figure(1);
set(fg,'WindowStyle','docked')
xlabel("Time [s]")
ylabel("Temperature [C°]")
hold on;

k = 1;
while true
    data_vect(k,:) = read(dq, 1, "OutputFormat", "Matrix");

    set(0,'CurrentFigure',fg);
    plot((k-1)/dq.Rate,data_vect(k,1),"rx");
    legend("Thermocouple K","Location","west");

    %ButtonHandle.Position = [.1 .1 .8 .8];

    if ~ishandle(fg)
        % Stop the if cancel button was pressed
        disp('Acquisition stopped');
        break;
    end

    pause(0.01); % A NEW LINE
    k = k+1;
end
close all
hold off
close all

fprintf("-------------------------------\n");

% acquisition vectors
time_vect = (0:1/dq.Rate:(length(data_vect)-1)/dq.Rate)';

% plot data
fg = figure(1);
set(fg,'WindowStyle','docked')
xlabel("Time [s]")
ylabel("Temperature [C°]")
plot(time_vect,data_vect(:,1),'rx');
legend("Thermocouple K","Location","west");

%% ---------------------------- SAVE DATA ------------------------------ %%
dataFold = "Data_DynamicCalibration";
if ~isfolder(dataFold)
    mkdir(dataFold);
end
saveName = string(datetime('now'),'uuuu-MM-dd''_''HH-mm-ss') + "_" + saveName + ".mat";
save(fullfile(dataFold,saveName),"time_vect","data_vect")
