clear; clc; close all;

%% --------------------------- VARIABLES ------------------------------- %%
% folder with saved acquisitions (data_vect only, no time_vect)
acqFolder = "C:\Users\loren\Desktop\Dati_vfinal\NewAcquisitions\Acquisitions";

% acquisition rate used when these files were recorded (see USB9126_Pt100Reading.m)
rate = 2; % [Hz]

% output folder for exported plots
outFolder = fullfile(fileparts(mfilename('fullpath')), "output");
if ~isfolder(outFolder)
    mkdir(outFolder);
end

%% --------------------------- LOAD + PLOT ------------------------------ %%
files = dir(fullfile(acqFolder, "*.mat"));

for i = 1:numel(files)
    matPath = fullfile(files(i).folder, files(i).name);
    S = load(matPath, "data_vect");
    data_vect = S.data_vect;
    time_vect = (0:1/rate:(length(data_vect)-1)/rate)';

    label = erase(files(i).name, ".mat");

    fgi = figure();
    set(fgi,'WindowStyle','docked')
    plot(time_vect, data_vect(:,1), "rx");
    xlabel("Time [s]")
    ylabel("Temperature [C°]")
    title(label, "Interpreter", "none")
    saveas(fgi, fullfile(outFolder, label + ".png"));
end

fprintf("Saved %d plots to %s\n", numel(files), outFolder);
