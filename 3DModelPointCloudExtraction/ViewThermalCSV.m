%% thermal_voxels.csv viewer with temperature colorbar
%
% Unlike ViewPCD.m (which reads .pcd/.ply and colors by height or with a
% fixed color), this script reads the CSV produced by
% voxel_consensus.py --stage thermal and colors each voxel by its mean
% corrected temperature (t_mean_c), with a real colorbar in degrees C.
%
% Expected columns in thermal_voxels.csv:
%   x, y, z, t_mean_c, t_std_c, n_obs, material, solar_absorptance
%
% REQUIREMENTS: no special toolbox, just scatter3 (Base MATLAB).

clear
close all
clc

%% 1. Parameters
csvPath = "C:\Users\loren\Desktop\Dati_vfinal\SLAM\ZED\20260730_161223\fullrate\voxel_map_m2f\thermal_voxels.csv";

% Optional Region Of Interest (ROI) filter, in metres, consistent with the
% frame aligned by aligned_octree.py. Leave [] to not filter.
roiXLimits = [12 Inf];   % e.g. [0 34]
roiYLimits = [-1.8 2];   % e.g. [-1 2]
roiZLimits = [];   % e.g. [0 2]

% Colormap and color range. [] = use the data's real min/max.
tempColormap = 'jet';
tempCLimits = [];   % e.g. [34 43], as in Figure 20

markerSize = 20;

%% 2. Loading
T = readtable(csvPath);
fprintf('Voxels loaded: %d\n', height(T));

%% 3. ROI filter (optional)
mask = true(height(T), 1);
if ~isempty(roiXLimits)
    mask = mask & T.x >= roiXLimits(1) & T.x <= roiXLimits(2);
end
if ~isempty(roiYLimits)
    mask = mask & T.y >= roiYLimits(1) & T.y <= roiYLimits(2);
end
if ~isempty(roiZLimits)
    mask = mask & T.z >= roiZLimits(1) & T.z <= roiZLimits(2);
end
T = T(mask, :);
fprintf('Voxels in ROI: %d\n', height(T));

%% 4. Visualization
figure('Color', 'w', 'Name', 'Thermal Voxels');
scatter3(T.x, T.y, T.z, markerSize, T.t_mean_c, 'filled');
axis equal
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
title(sprintf('%d voxels in ROI', height(T)));

colormap(gca, tempColormap);
if ~isempty(tempCLimits)
    clim(tempCLimits);
end
cb = colorbar;
cb.Label.String = 'Mean corrected temperature per voxel (\circC)';
