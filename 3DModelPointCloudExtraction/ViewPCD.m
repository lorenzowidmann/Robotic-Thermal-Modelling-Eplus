%% Visualizzatore semplice di un file .pcd
%
% Carica una nuvola di punti salvata (es. da SavedBag\...) e la mostra.
% Permette di alleggerirla prima della visualizzazione, utile su nuvole
% grandi/dense che altrimenti rendono pcshow lento:
%   - downsampling a voxel (gridAverage): riduce la densita' mediando i
%     punti dentro celle di lato voxelSize
%   - downsampling casuale: tiene solo una frazione dei punti, a caso
% I due si possono combinare (prima voxel, poi random).
%
% REQUISITI: Computer Vision Toolbox / Lidar Toolbox (pointCloud, pcshow)

clear
close all
clc

%% 1. Parametri
pcdPath =  "C:\Users\loren\Desktop\Measurment_v2\ClaudeCode\Thesis-final-wt2\3DModelPointCloudExtraction\SavedBag\merged_cloud.pcd";

% Downsampling a voxel: media i punti dentro celle cubiche di lato
% voxelSize. Riduce la densita' in modo uniforme nello spazio.
useVoxelDownsample = false;
voxelSize = 0.01;   % m

% Downsampling casuale: tiene solo densityFraction dei punti (0-1],
% scelti a caso. Utile per alleggerire una nuvola gia' uniforme senza
% cambiarne la risoluzione spaziale.
useRandomDownsample = false;
densityFraction = 0.5;   % 1.0 = nessuna riduzione, 0.1 = tiene 1 punto su 10

% Colore dei punti in pcshow: 'z' colora per quota (default), oppure
% un colore fisso tipo 'b'/[0 1 0].
colorBy = 'z';
markerSize = 20;

%% 2. Caricamento
pc = pcread(pcdPath);
fprintf('Nuvola caricata: %d punti\n', pc.Count);
fprintf('  X: %7.2f  %7.2f\n', pc.XLimits);
fprintf('  Y: %7.2f  %7.2f\n', pc.YLimits);
fprintf('  Z: %7.2f  %7.2f\n', pc.ZLimits);

%% 3. Riduzione densita' (opzionale)
if useVoxelDownsample
    nBefore = pc.Count;
    pc = pcdownsample(pc, 'gridAverage', voxelSize);
    fprintf('Downsampling voxel (%.3f m): %d -> %d punti\n', voxelSize, nBefore, pc.Count);
end

if useRandomDownsample
    nBefore = pc.Count;
    pc = pcdownsample(pc, 'random', densityFraction);
    fprintf('Downsampling casuale (%.0f%%): %d -> %d punti\n', 100*densityFraction, nBefore, pc.Count);
end

%% 4. Visualizzazione
figure('Color', 'k', 'Name', 'Visualizzatore PCD');
if strcmpi(colorBy, 'z')
    pcshow(pc, 'MarkerSize', markerSize);
else
    pcshow(pc.Location, colorBy, 'MarkerSize', markerSize);
end
title(sprintf('%s  (%d punti)', pcdPath, pc.Count), 'Color', 'w', 'Interpreter', 'none');
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
axis equal; colormap(gca, turbo);
