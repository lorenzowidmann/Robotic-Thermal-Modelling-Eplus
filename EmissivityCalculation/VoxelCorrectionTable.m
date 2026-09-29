function varargout = VoxelCorrectionTable(beforeCsv, afterCsv, opts)
%% Max / mean / min voxel temperature before and after radiometric correction
% Reads two (or three) thermal_voxels.csv files written by
% voxel_consensus.py --stage thermal and builds one row per case with the
% maximum, mean and minimum of t_mean_c (degrees C, one decimal), plus the
% number of voxels used. Same CSV layout as
% 3DModelPointCloudExtraction\ViewThermalCSV.m:
%   x, y, z, t_mean_c, t_std_c, n_obs, material, solar_absorptance
% NaN or empty t_mean_c cells are ignored (voxel_consensus.py already skips
% NaN pixels, so normally there are none).
%
% The table is printed, saved as CSV in the folder of beforeCsv, and printed
% again as a booktabs LaTeX tabular ready to paste.
%
% Name-value options:
%   EmissivityOnlyCsv  third CSV, emissivity correction only (adds a row, plus
%                      a Bias correction row = after - before - emissivity)
%   Roi                [xmin xmax ymin ymax zmin zmax] in the CSV frame (m),
%                      statistics only inside this box, e.g. around the RTD
%   ReferenceC         RTD / thermocouple temperature (C): adds a Reference
%                      row and a column with mean minus reference
%   OutputCsv          output path (default <folder of beforeCsv>\voxel_correction_table.csv)
%
% Usage:
%   VoxelCorrectionTable                            % default session (A1), below
%   VoxelCorrectionTable(beforeCsv, afterCsv)
%   VoxelCorrectionTable(beforeCsv, afterCsv, 'EmissivityOnlyCsv', emisCsv, ...
%       'Roi', [2.5 3.5 -3.8 -3.0 -0.5 1.5], 'ReferenceC', 22.19)

arguments
    beforeCsv (1,:) char = ''
    afterCsv (1,:) char = ''
    opts.EmissivityOnlyCsv (1,:) char = ''
    opts.Roi double = []
    opts.ReferenceC double = []
    opts.OutputCsv (1,:) char = ''
end

%% 0. Default session (only when called without inputs, e.g. Run button)
% A1, painted metal door, thermocouple 22.19 C. The voxel_map_* folders are
% written by voxel_consensus.py --stage thermal with --corrected-name
% apparent_temperature_masked.npy / corrected_temperature_debiased.npy /
% emissivity_correction.npy.
if nargin == 0
    sessionDir = 'C:\Users\loren\Desktop\Dati_vfinal\NewAcquisitions\AcquistionGroundTruth\Zed\20260911_094055\fullrate';
    beforeCsv = fullfile(sessionDir, 'voxel_map_before', 'thermal_voxels.csv');
    afterCsv  = fullfile(sessionDir, 'voxel_map_after_check', 'thermal_voxels.csv');
    opts.EmissivityOnlyCsv = fullfile(sessionDir, 'voxel_map_emis_corr', 'thermal_voxels.csv');
    opts.ReferenceC = 22.19;
elseif isempty(beforeCsv) || isempty(afterCsv)
    error('Pass both beforeCsv and afterCsv, or no input at all for the default session');
end

if ~isempty(opts.Roi) && numel(opts.Roi) ~= 6
    error('Roi must be [xmin xmax ymin ymax zmin zmax], got %d values', numel(opts.Roi));
end

%% 1. Cases to compare
paths = {beforeCsv, afterCsv};
names = {'Before correction', 'After correction'};
if ~isempty(opts.EmissivityOnlyCsv)
    paths{end+1} = opts.EmissivityOnlyCsv;
    names{end+1} = 'Emissivity correction only';
end

%% 2. Statistics per CSV
nCase = numel(paths);
tMax = nan(nCase, 1); tMean = nan(nCase, 1); tMin = nan(nCase, 1);
nVox = zeros(nCase, 1);
kept = cell(nCase, 1);   % voxels used per case, reused for the bias row
for k = 1:nCase
    if ~isfile(paths{k})
        error('CSV not found: %s', paths{k});
    end
    V = readtable(paths{k});
    t = V.t_mean_c;
    mask = isfinite(t);
    if ~isempty(opts.Roi)
        r = opts.Roi;
        mask = mask & V.x >= r(1) & V.x <= r(2) ...
                    & V.y >= r(3) & V.y <= r(4) ...
                    & V.z >= r(5) & V.z <= r(6);
    end
    kept{k} = V(mask, {'x', 'y', 'z', 't_mean_c'});
    t = t(mask);
    nVox(k) = numel(t);
    if nVox(k) == 0
        warning('%s: no valid voxel (check Roi)', names{k});
        continue
    end
    tMax(k) = max(t); tMean(k) = mean(t); tMin(k) = min(t);
    fprintf('%-28s %6d voxels used (of %d)  %s\n', names{k}, nVox(k), height(V), paths{k});
end

%% 2b. Bias (sensor offset) correction per voxel
% All maps sample the same pixels and voxel averaging is linear, so per voxel
%   after = before + emissivity correction + bias correction.
% The bias correction is therefore after - before - emissivity, matched by
% voxel centre. Only possible when the emissivity-only CSV is given.
if nCase == 3
    keyed = cell(3, 1);
    for k = 1:3
        K = kept{k};
        % Voxel centres are float multiples of the voxel size: round to 1 um
        % so the same voxel matches across files.
        K.kx = round(K.x * 1e6); K.ky = round(K.y * 1e6); K.kz = round(K.z * 1e6);
        K.Properties.VariableNames{'t_mean_c'} = sprintf('t%d', k);
        keyed{k} = K(:, {'kx', 'ky', 'kz', sprintf('t%d', k)});
    end
    J = innerjoin(innerjoin(keyed{1}, keyed{2}), keyed{3});
    b = J.t2 - J.t1 - J.t3;
    names{end+1} = 'Bias correction';
    nVox(end+1) = numel(b);
    if isempty(b)
        tMax(end+1) = NaN; tMean(end+1) = NaN; tMin(end+1) = NaN;
        warning('Bias correction: no voxel common to the three CSVs');
    else
        tMax(end+1) = max(b); tMean(end+1) = mean(b); tMin(end+1) = min(b);
    end
    fprintf('%-28s %6d voxels used (after - before - emissivity)\n', names{end}, nVox(end));
end

%% 3. Table
% Reference row keeps only the mean, max and min are left empty (NaN).
hasRef = ~isempty(opts.ReferenceC);
if hasRef
    names{end+1} = 'Reference';
    tMax(end+1) = NaN; tMean(end+1) = opts.ReferenceC; tMin(end+1) = NaN;
    nVox(end+1) = NaN;
end

T = table(round(tMax, 1), round(tMean, 1), round(tMin, 1), nVox, ...
          'VariableNames', {'Max', 'Mean', 'Min', 'NVoxels'}, 'RowNames', names);
if hasRef
    % Difference from the unrounded mean, then rounded.
    T.DeltaMean = round(tMean - opts.ReferenceC, 1);
    T.DeltaMean(end) = NaN;
    % Correction rows hold a delta T, not a temperature.
    T.DeltaMean(ismember(names, {'Emissivity correction only', 'Bias correction'})) = NaN;
end

fprintf('\nVoxel temperature (C)');
if ~isempty(opts.Roi)
    fprintf(', ROI %s', mat2str(opts.Roi));
end
fprintf('\n');
disp(T);

%% 4. Save CSV next to the inputs
outCsv = opts.OutputCsv;
if isempty(outCsv)
    outCsv = fullfile(fileparts(beforeCsv), 'voxel_correction_table.csv');
end
writetable(T, outCsv, 'WriteRowNames', true);
fprintf('Saved: %s\n\n', outCsv);

%% 5. LaTeX tabular (booktabs)
if hasRef
    fprintf('\\begin{tabular}{lrrrr}\n\\toprule\n');
    fprintf('Case & Max (\\si{\\celsius}) & Mean (\\si{\\celsius}) & Min (\\si{\\celsius}) & $\\Delta$ Mean (\\si{\\celsius}) \\\\\n');
else
    fprintf('\\begin{tabular}{lrrr}\n\\toprule\n');
    fprintf('Case & Max (\\si{\\celsius}) & Mean (\\si{\\celsius}) & Min (\\si{\\celsius}) \\\\\n');
end
fprintf('\\midrule\n');
for k = 1:height(T)
    if hasRef && k == height(T)
        fprintf('\\midrule\n');
    end
    cells = {fmt(T.Max(k)), fmt(T.Mean(k)), fmt(T.Min(k))};
    if hasRef
        cells{end+1} = fmt(T.DeltaMean(k));
    end
    fprintf('%s & %s \\\\\n', names{k}, strjoin(cells, ' & '));
end
fprintf('\\bottomrule\n\\end{tabular}\n');

% Return the table only if asked (T = VoxelCorrectionTable(...)), so a plain
% call does not print it a second time as "ans".
if nargout > 0
    varargout{1} = T;
end
end

function s = fmt(v)
% One decimal, empty cells as "--" (LaTeX en-dash).
if isnan(v)
    s = '--';
else
    s = sprintf('%.1f', v);
end
end
