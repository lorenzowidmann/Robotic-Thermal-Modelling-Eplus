function ShowTemperatureCorrection(sessionDirIn, flirDirIn)
%% Interactive comparison of apparent / corrected temperature (session 9)
% Shows the session's FLIR frame and, hovering the mouse over a pixel,
% reads the temperature BEFORE radiometric correction (the raw camera .npy,
% apparent temperature) and AFTER (corrected_temperature_consensus.npy
% produced by RadiometricCalibration\correct_session.py with the multi-view
% consensus materials from EmissivityCalculation\voxel_consensus.py --stage
% vote), with the difference.
%
% Same idea as RadiometricCalibration\ThermalData.py --show (hover to read
% the value under the cursor), but here there are two values side by side,
% and the whole session can be scrolled through with the arrow keys.
%
% Besides the temperature, the pointed pixel also shows the data that
% determined the correction: applied emissivity, LiDAR distance, consensus
% material (with the vote's agreement, and whether the consensus changed its
% mind relative to the single view), and whether that pixel was a direct
% LiDAR sample or filled by nearest neighbour (sampled_mask). This makes it
% immediately clear whether a strange value is measured or interpolated.
%
% Keyboard commands:
%   right / left arrow    next / previous frame
%   page up / page down   forward / back 10 frames
%   home / end            first / last frame
%   c                      lock (or unlock) the reading on the clicked point
%   s                      show / hide the direct LiDAR samples
%   b                      show / hide the superpixel boundaries (segment_id)
%   l                      fixed colour scale over the whole session / per frame
%
% Usage:
%   ShowTemperatureCorrection                       % default paths, below
%   ShowTemperatureCorrection(sessionDir)           % another ZED session
%   ShowTemperatureCorrection(sessionDir, flirDir)

close all
clc

%% 1. Parameters
sessionDir = 'C:\Users\loren\Desktop\Dati_vfinal\SLAM\ZED\20260730_161223\fullrate';
flirDir    = 'C:\Users\loren\Desktop\Dati_vfinal\SLAM\Flir\session9_only_rot180';

% Name of the corrected file to read in each emissivity_map\<stem>\, and the
% materials folder the readout text ("segment N = material") comes from.
% Defaults are the ones produced by the multi-view consensus (the current
% pipeline): correctedName in emissivity_map\<stem>\ is written by
% correct_session.py --material-map-dir material_map_consensus, while
% emissivity_used.npy / correction_report.json have no configurable name and
% are always overwritten by the last correct_session.py run -- so they must
% already be read together with this file, not with the old
% corrected_temperature.npy (which on disk is still the very first run,
% single-view, and is no longer consistent with emissivity_used).
correctedName = 'corrected_temperature_consensus.npy';
materialDirName = 'material_map_consensus';

% The arguments, if passed, take precedence over the defaults above.
if nargin >= 1 && ~isempty(sessionDirIn)
    sessionDir = sessionDirIn;
end
if nargin >= 2 && ~isempty(flirDirIn)
    flirDir = flirDirIn;
end

emisDir     = fullfile(sessionDir, 'emissivity_map');
materialDir = fullfile(sessionDir, materialDirName);

if ~isfolder(emisDir)
    error('emissivity_map folder not found: %s', emisDir);
end
if ~isfolder(flirDir)
    error('FLIR folder not found: %s', flirDir);
end
if ~isfolder(materialDir)
    warning('Materials folder not found: %s (the readout will not show materials)', materialDir);
end

%% 2. List of corrected frames
% A frame is usable only if both the apparent .npy and the correction exist:
% correct_session.py skips frames without distance/segment_id.
d = dir(fullfile(emisDir, '*'));
d = d([d.isdir] & ~ismember({d.name}, {'.', '..'}));

frames = struct('stem', {}, 'apparentPath', {}, 'correctedPath', {}, 'dir', {});
for k = 1:numel(d)
    stem = d(k).name;                       % e.g. 20250906_233144_R
    frameDir = fullfile(emisDir, stem);
    corrPath = fullfile(frameDir, correctedName);

    % The raw FLIR .npy does not have the folder name's _R suffix.
    appPath = fullfile(flirDir, [strrep(stem, '_R', '') '.npy']);

    if isfile(corrPath) && isfile(appPath)
        frames(end+1) = struct('stem', stem, ...
                               'apparentPath', appPath, ...
                               'correctedPath', corrPath, ...
                               'dir', frameDir); %#ok<AGROW>
    end
end

if isempty(frames)
    error('No frame with "%s" found in %s', correctedName, emisDir);
end
nFrames = numel(frames);
fprintf('%d corrected frames (%s) in %s\n', nFrames, correctedName, emisDir);

%% 3. Colour scale common to the whole session
% Taken from correction_report.json, so there is no need to re-read every
% .npy: a fixed scale makes the comparison between frames honest (a frame
% does not look hotter just because it was rescaled against itself). The
% report has no configurable name and is always the one from the last
% correct_session.py run, so it is automatically consistent with
% correctedName above as long as that run was made with the same
% --material-map-dir as the consensus.
%
% The absolute minimum and maximum are not used, though: in this session a
% single frame reaches 62 C on a handful of pixels, and taking that
% literally would squash every other frame into the first fifth of the
% palette (all-dark-purple images). The tail is therefore discarded using
% percentiles over each frame's own min/max: the scale stays common to the
% whole session, but genuinely covered by the data.
fMin = nan(nFrames, 1);
fMax = nan(nFrames, 1);
for k = 1:nFrames
    repPath = fullfile(frames(k).dir, 'correction_report.json');
    if ~isfile(repPath), continue; end
    rep = jsondecode(fileread(repPath));
    if ~isempty(rep.corrected_c.min)
        fMin(k) = rep.corrected_c.min;
        fMax(k) = rep.corrected_c.max;
    end
end
fMin = fMin(~isnan(fMin));
fMax = fMax(~isnan(fMax));
if isempty(fMin)
    sessMin = 20; sessMax = 50;
else
    sessMin = pctOf(fMin, 0.05);
    sessMax = pctOf(fMax, 0.95);
end
fprintf('Session colour scale: %.1f .. %.1f C', sessMin, sessMax);
if ~isempty(fMax)
    fprintf('  (real extremes %.1f .. %.1f C, tails excluded)', min(fMin), max(fMax));
end
fprintf('\n');

%% 4. Interface state
S.frames      = frames;
S.nFrames     = nFrames;
S.idx         = 1;
S.sessClim    = [sessMin sessMax];
S.lockedClim  = true;    % true = fixed session scale, false = per frame
S.showSamples = false;   % overlay of the direct LiDAR samples
S.showSegs    = true;    % overlay of the superpixel boundaries (SLIC grid)
S.pinned      = false;   % reading locked on the clicked point
S.pinXY       = [NaN NaN];
S.materialDir = materialDir;
S.cache       = struct();

%% 5. Figure
S.fig = figure('Name', 'Radiometric correction - apparent vs corrected (consensus)', ...
               'NumberTitle', 'off', 'Color', 'w', ...
               'Units', 'normalized', 'Position', [0.08 0.15 0.84 0.70]);

S.axApp  = subplot(1, 2, 1, 'Parent', S.fig);
S.axCorr = subplot(1, 2, 2, 'Parent', S.fig);

S.imApp  = imagesc(S.axApp,  zeros(2));
S.imCorr = imagesc(S.axCorr, zeros(2));
axis(S.axApp,  'image'); axis(S.axCorr, 'image');
colormap(S.fig, inferno_like());
cb1 = colorbar(S.axApp);  cb1.Label.String = 'deg C';
cb2 = colorbar(S.axCorr); cb2.Label.String = 'deg C';

hold(S.axApp,  'on');
hold(S.axCorr, 'on');
S.samplesApp  = plot(S.axApp,  NaN, NaN, '.', 'Color', [0 0.6 1], 'MarkerSize', 1);
S.samplesCorr = plot(S.axCorr, NaN, NaN, '.', 'Color', [0 0.6 1], 'MarkerSize', 1);
% Superpixel grid: a single NaN-separated line per axis.
S.segsApp  = plot(S.axApp,  NaN, NaN, '-', 'Color', [1 1 1 0.45], 'LineWidth', 0.5);
S.segsCorr = plot(S.axCorr, NaN, NaN, '-', 'Color', [1 1 1 0.45], 'LineWidth', 0.5);
S.markApp  = plot(S.axApp,  NaN, NaN, '+', 'Color', 'c', 'MarkerSize', 14, 'LineWidth', 1.5);
S.markCorr = plot(S.axCorr, NaN, NaN, '+', 'Color', 'c', 'MarkerSize', 14, 'LineWidth', 1.5);
hold(S.axApp,  'off');
hold(S.axCorr, 'off');

% Readout line under the two images.
S.readout = annotation(S.fig, 'textbox', [0.02 0.005 0.96 0.085], ...
                       'String', '', 'FontName', 'Consolas', 'FontSize', 10, ...
                       'EdgeColor', [0.8 0.8 0.8], 'BackgroundColor', [0.97 0.97 0.97], ...
                       'VerticalAlignment', 'middle', 'Interpreter', 'none');

guidata(S.fig, S);
set(S.fig, 'WindowKeyPressFcn',        @(src, ev) onKey(src, ev), ...
           'WindowButtonMotionFcn',    @(src, ev) onMove(src), ...
           'WindowButtonDownFcn',      @(src, ev) onClick(src));

loadFrame(S.fig, 1);

end % ShowTemperatureCorrection


%% ------------------------------------------------------------------ %%
function loadFrame(fig, idx)
%LOADFRAME Loads and draws frame idx (with a cache of already-read .npy files).
S = guidata(fig);
S.idx = max(1, min(S.nFrames, idx));
f = S.frames(S.idx);

key = matlab.lang.makeValidName(f.stem);
if ~isfield(S.cache, key)
    D.apparent  = readNPY(f.apparentPath);
    D.corrected = readNPY(f.correctedPath);

    % The accessory files may be missing: reading is still possible, just
    % without emissivity / distance / material.
    D.emissivity = tryReadNPY(fullfile(f.dir, 'emissivity_used.npy'));
    D.distance   = tryReadNPY(fullfile(f.dir, 'distance.npy'));
    D.segment    = tryReadNPY(fullfile(f.dir, 'segment_id.npy'));
    D.sampled    = tryReadNPY(fullfile(f.dir, 'sampled_mask.npy'));

    % The superpixel boundaries are computed once: they are fixed per frame.
    [D.segX, D.segY] = segBoundaryLines(D.segment);

    % Consensus segments.json: top_material is the FINAL material (after the
    % vote), consensus.from_frame is the single view's choice if the vote
    % changed it, consensus.agreement is the fraction of votes backing the
    % final material.
    D.material = containers.Map('KeyType', 'double', 'ValueType', 'any');
    segPath = fullfile(S.materialDir, f.stem, 'segments.json');
    if isfile(segPath)
        seg = jsondecode(fileread(segPath));
        for i = 1:numel(seg.segments)
            s = seg.segments(i);
            if iscell(s), s = s{1}; end
            D.material(double(s.id)) = s;
        end
    end

    S.cache.(key) = D;
end
D = S.cache.(key);

set(S.imApp,  'CData', D.apparent);
set(S.imCorr, 'CData', D.corrected);
[h, w] = size(D.apparent);
set(S.axApp,  'XLim', [0.5 w+0.5], 'YLim', [0.5 h+0.5]);
set(S.axCorr, 'XLim', [0.5 w+0.5], 'YLim', [0.5 h+0.5]);

if S.lockedClim
    clim(S.axApp,  S.sessClim);
    clim(S.axCorr, S.sessClim);
    climTag = sprintf('fixed scale %.1f-%.1f C', S.sessClim(1), S.sessClim(2));
else
    clim(S.axApp,  robustClim(D.apparent));
    clim(S.axCorr, robustClim(D.corrected));
    climTag = 'per-frame scale';
end

title(S.axApp, sprintf('BEFORE - apparent (raw FLIR)\nmin %.1f  max %.1f  mean %.1f C', ...
      min(D.apparent(:)), max(D.apparent(:)), mean(D.apparent(:))), 'FontSize', 9);
title(S.axCorr, sprintf('AFTER - corrected (multi-view consensus)\nmin %.1f  max %.1f  mean %.1f C', ...
      min(D.corrected(:), [], 'omitnan'), max(D.corrected(:), [], 'omitnan'), ...
      mean(D.corrected(:), 'omitnan')), 'FontSize', 9);

nNaN = sum(isnan(D.corrected(:)));
sgtitle(S.fig, sprintf('[%d/%d]  %s     %s     NaN %d px     (arrows = scroll, click = lock point, b = superpixel boundaries)', ...
        S.idx, S.nFrames, f.stem, climTag, nNaN), 'FontSize', 10, 'Interpreter', 'none');

if S.showSamples && ~isempty(D.sampled)
    [ys, xs] = find(D.sampled);
    set(S.samplesApp,  'XData', xs, 'YData', ys);
    set(S.samplesCorr, 'XData', xs, 'YData', ys);
else
    set(S.samplesApp,  'XData', NaN, 'YData', NaN);
    set(S.samplesCorr, 'XData', NaN, 'YData', NaN);
end

if S.showSegs && ~isempty(D.segX)
    set(S.segsApp,  'XData', D.segX, 'YData', D.segY);
    set(S.segsCorr, 'XData', D.segX, 'YData', D.segY);
else
    set(S.segsApp,  'XData', NaN, 'YData', NaN);
    set(S.segsCorr, 'XData', NaN, 'YData', NaN);
end

guidata(fig, S);
if S.pinned
    updateReadout(fig, S.pinXY(1), S.pinXY(2));
else
    updateReadout(fig, NaN, NaN);
end
end


%% ------------------------------------------------------------------ %%
function updateReadout(fig, xi, yi)
%UPDATEREADOUT Fills the text line with the values of pixel (xi, yi).
S = guidata(fig);
D = S.cache.(matlab.lang.makeValidName(S.frames(S.idx).stem));
[h, w] = size(D.apparent);

if isnan(xi) || xi < 1 || xi > w || yi < 1 || yi > h
    set(S.readout, 'String', ...
        'Point the mouse over the image to read the temperature before / after the correction.');
    set(S.markApp,  'XData', NaN, 'YData', NaN);
    set(S.markCorr, 'XData', NaN, 'YData', NaN);
    return
end

tBefore = double(D.apparent(yi, xi));
tAfter  = double(D.corrected(yi, xi));
delta   = tAfter - tBefore;

if isnan(tAfter)
    line1 = sprintf('x=%3d y=%3d   BEFORE %6.2f C   AFTER   NaN (no plausible material)', ...
                    xi, yi, tBefore);
else
    line1 = sprintf('x=%3d y=%3d   BEFORE %6.2f C   AFTER %6.2f C   DELTA %+5.2f C', ...
                    xi, yi, tBefore, tAfter, delta);
end

% Second line: where that correction comes from.
parts = {};
if ~isempty(D.emissivity) && ~isnan(D.emissivity(yi, xi))
    parts{end+1} = sprintf('emissivity %.3f', D.emissivity(yi, xi));
end
if ~isempty(D.distance) && D.distance(yi, xi) > 0
    parts{end+1} = sprintf('LiDAR distance %.2f m', D.distance(yi, xi));
end
if ~isempty(D.segment)
    sid = double(D.segment(yi, xi));
    if sid >= 0 && isKey(D.material, sid)
        s = D.material(sid);
        % consensus.status: 'ok' = the segment received votes and shows the
        % vote's agreement on the FINAL material; 'no_lidar_sample' = no
        % LiDAR point from any view landed on this segment, so the material
        % is still the single view's own (pure CLIP).
        hasConsensus = isfield(s, 'consensus') && isstruct(s.consensus) && isfield(s.consensus, 'status');
        if hasConsensus && strcmp(s.consensus.status, 'ok')
            c = s.consensus;
            if isfield(c, 'from_frame') && ~isempty(c.from_frame) && ~strcmp(c.from_frame, s.top_material)
                parts{end+1} = sprintf('segment %d = %s (vote %.0f%%, the single view said %s)', ...
                                       sid, s.top_material, 100 * c.agreement, c.from_frame);
            else
                parts{end+1} = sprintf('segment %d = %s (vote %.0f%%)', ...
                                       sid, s.top_material, 100 * c.agreement);
            end
        else
            parts{end+1} = sprintf('segment %d = %s (CLIP %.0f%%, no multi-view consensus)', ...
                                   sid, s.top_material, 100 * s.confidence);
        end
    else
        parts{end+1} = sprintf('segment %d', sid);
    end
end
if ~isempty(D.sampled)
    if D.sampled(yi, xi)
        parts{end+1} = 'direct LiDAR sample';
    else
        parts{end+1} = 'filled by nearest neighbour (not measured)';
    end
end
line2 = strjoin(parts, '   |   ');

if S.pinned
    line1 = ['[LOCKED]  ' line1];
end
set(S.readout, 'String', {line1, line2});
set(S.markApp,  'XData', xi, 'YData', yi);
set(S.markCorr, 'XData', xi, 'YData', yi);
end


%% ------------------------------------------------------------------ %%
function [xi, yi, inside] = cursorPixel(fig)
%CURSORPIXEL Integer pixel under the cursor, in either of the two axes.
S = guidata(fig);
xi = NaN; yi = NaN; inside = false;
for ax = [S.axApp, S.axCorr]
    p = get(ax, 'CurrentPoint');
    x = round(p(1, 1));
    y = round(p(1, 2));
    xl = get(ax, 'XLim'); yl = get(ax, 'YLim');
    if x >= xl(1) && x <= xl(2) && y >= yl(1) && y <= yl(2)
        xi = x; yi = y; inside = true;
        return
    end
end
end


function onMove(fig)
S = guidata(fig);
if S.pinned, return; end
[xi, yi] = cursorPixel(fig);
updateReadout(fig, xi, yi);
end


function onClick(fig)
S = guidata(fig);
[xi, yi, inside] = cursorPixel(fig);
if ~inside, return; end
S.pinned = true;
S.pinXY = [xi yi];
guidata(fig, S);
updateReadout(fig, xi, yi);
end


function onKey(fig, ev)
S = guidata(fig);
switch ev.Key
    case 'rightarrow', loadFrame(fig, S.idx + 1);
    case 'leftarrow',  loadFrame(fig, S.idx - 1);
    case 'pagedown',   loadFrame(fig, S.idx + 10);
    case 'pageup',     loadFrame(fig, S.idx - 10);
    case 'home',       loadFrame(fig, 1);
    case 'end',        loadFrame(fig, S.nFrames);
    case 'c'
        S.pinned = ~S.pinned;
        guidata(fig, S);
        if S.pinned
            updateReadout(fig, S.pinXY(1), S.pinXY(2));
        else
            updateReadout(fig, NaN, NaN);
        end
    case 's'
        S.showSamples = ~S.showSamples;
        guidata(fig, S);
        loadFrame(fig, S.idx);
    case 'b'
        S.showSegs = ~S.showSegs;
        guidata(fig, S);
        loadFrame(fig, S.idx);
    case 'l'
        S.lockedClim = ~S.lockedClim;
        guidata(fig, S);
        loadFrame(fig, S.idx);
end
end


%% ------------------------------------------------------------------ %%
function [X, Y] = segBoundaryLines(seg)
%SEGBOUNDARYLINES Superpixel grid: NaN-separated line segments along every
% boundary between two different segment_id values. The lines fall halfway
% between two pixels (x+0.5 / y+0.5), so they line up with the edges shown
% by imagesc and draw the SLIC squares without covering the pixels.
X = NaN; Y = NaN;
if isempty(seg)
    return
end
seg = double(seg);

% Vertical boundaries: between column x and x+1.
[yv, xv] = find(seg(:, 1:end-1) ~= seg(:, 2:end));
Xv = [xv + 0.5, xv + 0.5, nan(size(xv))]';
Yv = [yv - 0.5, yv + 0.5, nan(size(yv))]';

% Horizontal boundaries: between row y and y+1.
[yh, xh] = find(seg(1:end-1, :) ~= seg(2:end, :));
Xh = [xh - 0.5, xh + 0.5, nan(size(xh))]';
Yh = [yh + 0.5, yh + 0.5, nan(size(yh))]';

X = [Xv(:); Xh(:)];
Y = [Yv(:); Yh(:)];
if isempty(X)
    X = NaN; Y = NaN;
end
end


%% ------------------------------------------------------------------ %%
function c = robustClim(A)
%ROBUSTCLIM Colour limits on the 1-99 percentiles, so a single hot pixel
% does not squash the rest of the image.
v = A(isfinite(A));
if isempty(v)
    c = [0 1];
    return
end
c = [pctOf(v, 0.01) pctOf(v, 0.99)];
if c(2) <= c(1)
    c = [min(v) max(v) + eps];
end
end


function p = pctOf(v, q)
%PCTOF Percentile q (0-1) of v, computed by hand to avoid depending on the
% Statistics Toolbox (prctile is not always available).
v = sort(v(isfinite(v)));
if isempty(v)
    p = NaN;
    return
end
p = v(min(numel(v), max(1, round(q * numel(v)))));
end


function cmap = inferno_like()
%INFERNO_LIKE Dark->yellow palette in the style of matplotlib's 'inferno',
% to match the look of ThermalData.py --show without extra toolboxes.
anchors = [0.001 0.000 0.014
           0.259 0.039 0.406
           0.576 0.149 0.404
           0.865 0.317 0.226
           0.988 0.645 0.040
           0.988 0.998 0.645];
x = linspace(0, 1, size(anchors, 1));
xi = linspace(0, 1, 256);
cmap = [interp1(x, anchors(:,1), xi)', ...
        interp1(x, anchors(:,2), xi)', ...
        interp1(x, anchors(:,3), xi)'];
end


%% ------------------------------------------------------------------ %%
function A = tryReadNPY(path)
%TRYREADNPY Like readNPY but returns [] if the file does not exist.
if isfile(path)
    A = readNPY(path);
else
    A = [];
end
end


function A = readNPY(path)
%READNPY Minimal reader for NumPy's .npy format (v1.0/2.0).
% Covers only the cases produced by this pipeline: 2-D array, little-endian,
% C order, dtype float32/float64/int32/int64/uint8/bool. Avoids depending on
% npy-matlab, which is not installed.
fid = fopen(path, 'r');
if fid < 0
    error('Could not open %s', path);
end
cleaner = onCleanup(@() fclose(fid));

magic = fread(fid, 6, '*uint8')';
if ~isequal(magic, uint8([147 78 85 77 80 89]))   % \x93NUMPY
    error('%s is not a .npy file', path);
end
major = fread(fid, 1, 'uint8');
fread(fid, 1, 'uint8');                            % minor, not needed
if major == 1
    headerLen = fread(fid, 1, 'uint16=>double');
else
    headerLen = fread(fid, 1, 'uint32=>double');
end
header = fread(fid, headerLen, '*char')';

descr = regexp(header, '''descr''\s*:\s*''([^'']+)''', 'tokens', 'once');
fortran = regexp(header, '''fortran_order''\s*:\s*(True|False)', 'tokens', 'once');
shapeTok = regexp(header, '''shape''\s*:\s*\(([^)]*)\)', 'tokens', 'once');
if isempty(descr) || isempty(shapeTok)
    error('Unrecognised .npy header in %s', path);
end
descr = descr{1};
shape = str2double(strsplit(strtrim(strrep(shapeTok{1}, ',', ' ')), ' '));
shape = shape(~isnan(shape));

switch descr
    case {'<f4', '=f4', 'f4'},                 fmt = 'single=>single';
    case {'<f8', '=f8', 'f8'},                 fmt = 'double=>double';
    case {'<i4', '=i4', 'i4'},                 fmt = 'int32=>int32';
    case {'<i8', '=i8', 'i8'},                 fmt = 'int64=>int64';
    case {'|u1', '<u1', 'u1'},                 fmt = 'uint8=>uint8';
    case {'|b1', '<b1', 'b1'},                 fmt = 'uint8=>uint8';
    otherwise
        error('Unsupported .npy dtype (%s) in %s', descr, path);
end

n = prod(shape);
raw = fread(fid, n, fmt, 0, 'ieee-le');
if numel(raw) ~= n
    error('Truncated .npy file: %s', path);
end

% NumPy stores in C order (consecutive rows), MATLAB reads in column order:
% it is filled transposed and then transposed back.
if strcmp(fortran{1}, 'True')
    A = reshape(raw, shape);
else
    A = reshape(raw, fliplr(shape))';
end
if strcmp(descr(end-1:end), 'b1')
    A = logical(A);
end
if ~islogical(A) && ~isinteger(A)
    A = double(A);
end
end
