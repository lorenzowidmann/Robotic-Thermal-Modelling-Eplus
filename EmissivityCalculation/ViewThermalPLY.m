%% Point cloud viewer for the thermal voxel map (thermal_voxels.ply)
%
% Same idea as 3DModelPointCloudExtraction\ViewPCD.m -- load a point cloud,
% optionally thin it out, show it -- but for the OUTPUT of
% EmissivityCalculation\voxel_consensus.py rather than a raw SLAM cloud.
%
% thermal_voxels.ply has no separate "temperature" field: voxel_consensus.py
% bakes the corrected temperature straight into the point RGB, blue->red over
% the session's 5th-95th percentile (see the file header, and the .py's
% "Same data as a PLY..." comment). So colouring by Z like ViewPCD.m does by
% default would show the room's shape and throw away the only reason this
% file exists. That is why colorBy here defaults to 'rgb' -- use the file's
% own colours -- rather than 'z'.
%
% REQUIRES: Computer Vision Toolbox / Lidar Toolbox (pointCloud, pcshow)

clear
close all
clc

%% 1. Parameters
plyPath = 'C:\Users\loren\Desktop\Dati_vfinal\SLAM\ZED\20260730_161223\fullrate\voxel_map\thermal_voxels.ply';

% Voxel downsampling: averages points inside voxelSize cubes. Reduces
% density uniformly in space. Off by default -- the cloud is already one
% point per voxel_consensus.py voxel (args.voxel, typically 0.2 m), so this
% would coarsen a grid that was already deliberately chosen.
useVoxelDownsample = false;
voxelSize = 0.05;   % m

% Random downsampling: keeps only densityFraction of the points, chosen at
% random. Useful to lighten an already-uniform cloud without changing its
% spatial resolution.
useRandomDownsample = false;
densityFraction = 0.5;   % 1.0 = no reduction, 0.1 = keeps 1 point in 10

% Point colour in pcshow:
%   'rgb'  (default) -- the file's own colours, i.e. corrected temperature
%          blue (cold, 5th percentile) to red (hot, 95th percentile).
%   'z'    -- colour by height instead, like ViewPCD.m's default. Loses the
%             temperature encoding; use only to check the cloud's shape.
%   anything else (e.g. 'b', [0 1 0]) -- a single fixed colour.
colorBy = 'rgb';
markerSize = 30;

%% 2. Loading
pc = pcread(plyPath);
fprintf('Cloud loaded: %d voxels\n', pc.Count);
fprintf('  X: %7.2f  %7.2f\n', pc.XLimits);
fprintf('  Y: %7.2f  %7.2f\n', pc.YLimits);
fprintf('  Z: %7.2f  %7.2f\n', pc.ZLimits);
if isempty(pc.Color)
    warning(['%s has no RGB data -- colorBy=''rgb'' will fall back to Z. ' ...
            'Was this file really written by voxel_consensus.py --stage thermal?'], plyPath);
    colorBy = 'z';
end

%% 3. Density reduction (optional)
% pcdownsample keeps Color in sync with the surviving points either way, so
% the temperature encoding is still valid after thinning.
if useVoxelDownsample
    nBefore = pc.Count;
    pc = pcdownsample(pc, 'gridAverage', voxelSize);
    fprintf('Voxel downsampling (%.3f m): %d -> %d points\n', voxelSize, nBefore, pc.Count);
end

if useRandomDownsample
    nBefore = pc.Count;
    pc = pcdownsample(pc, 'random', densityFraction);
    fprintf('Random downsampling (%.0f%%): %d -> %d points\n', 100 * densityFraction, nBefore, pc.Count);
end

%% 4. Display
figure('Color', 'k', 'Name', 'Thermal voxel map viewer');
if strcmpi(colorBy, 'rgb')
    pcshow(pc, 'MarkerSize', markerSize);
    cTag = 'colour = corrected temperature (blue = cold .. red = hot, 5-95 pct)';
elseif strcmpi(colorBy, 'z')
    pcshow(pc.Location, pc.Location(:, 3), 'MarkerSize', markerSize);
    colormap(gca, turbo);
    cTag = 'colour = height (Z), NOT temperature';
else
    pcshow(pc.Location, colorBy, 'MarkerSize', markerSize);
    cTag = sprintf('colour = fixed (%s)', mat2str(colorBy));
end
title(sprintf('%s\n%d voxels -- %s', plyPath, pc.Count, cTag), ...
     'Color', 'w', 'Interpreter', 'none');
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
axis equal;
