function FlirLidarZedViewer(startIdx)
%FLIRLIDARZEDVIEWER Interactive FLIR->ZED overlay, session 9, steppable pose by pose.
%
% For the details and the sources of the parameters (intrinsics, extrinsics,
% FLIR rot180 convention) see README.md in this folder. Navigation: each keypress
% loads the next/previous TRIPLET from the sync manifest -> new LiDAR scan from
% the bag, new FLIR image, new ZED frame, new projection.
%
% Usage:
%   FlirLidarZedViewer        % starts at pose 9
%   FlirLidarZedViewer(30)    % starts at pose 30
%
% Keys (the figure window must have focus):
%   right arrow / n      -> next pose
%   left arrow / p       -> previous pose
%   s                    -> save the current frame to output/
%   q / close window     -> quit
%
% NOTE: requires an interactive MATLAB session (desktop). It does not work under
% `matlab -batch`, because in batch the window closes as soon as the script
% returns and no event loop is left to listen for keys.

close all
clc
% NB: no "clear" here - inside a function it would also wipe the input
% argument startIdx before it is even read (a function call starts with a
% clean workspace anyway, so it is not needed).

if nargin < 1
    startIdx = 9;
end

%% Fixed parameters (sources documented in README.md)

sessionRoot   = 'C:\Users\loren\Desktop\Dati_vfinal\SLAM';
S.zedSessionDir = fullfile(sessionRoot, 'ZED', '20260730_161223', 'fullrate');
S.flirRot180Dir = fullfile(sessionRoot, 'Flir', 'session9_only_rot180');
bagPath       = fullfile(sessionRoot, 'Lidar', 'rosbag2_2026_07_30-18_12_20');
syncManifestPath = fullfile(S.zedSessionDir, 'sync_manifest.json');

% The Livox HAP scans non-repetitively: ONE single scan (~0.2s, the spacing
% between /cloud_registered messages) only covers partial bands of the scene,
% hence the wide stripes visible on the overlay. To densify it, several
% consecutive scans around the chosen pose are merged (all transformed
% world->body with the SAME /Odometry pose of the triplet, so the rig is
% assumed near-stationary over the window: wider = more points but more
% "motion blur" for surfaces moving in the scene).
%
% Keep this at ONE scan. /cloud_registered is 5 Hz, so 0.4 s merges 4 scans
% spanning 0.8 s = 56 cm of travel at the 0.7 m/s of the end of session 9.
% The extra points are in the right place in the world, but they were observed
% from up to 28 cm away, so they include surfaces that are OCCLUDED from the
% pose being rendered. The 8 cm z-buffer cannot reject them, they take the
% thermal value of whatever foreground surface they land behind, and the hot
% pattern bleeds sideways off the structure it belongs to - tens of cm at the
% end of the session, nothing at all while standing still. That is what looked
% like accumulating drift after pose ~65.
S.lidarAccumHalfWindow_s = 0.1;   % seconds BEFORE and AFTER the pose timestamp

% Per-camera occlusion filter (z-buffer): FLIR and ZED are not co-located
% (~13cm real baseline between them on the rig), so near an edge the two
% cameras see "around the corner" differently. Without this filter, a point on
% a wall hidden behind an edge (invisible to the FLIR but geometrically inside
% its frustum) still gets sampled, wrongly picking up the colour of the
% foreground edge. For each (integer) pixel of each camera only the nearest
% point is kept (+- tol).
S.zBufferTol_m = 0.08;   % margin beyond the nearest, same order as the calibration RMSE
S.outDir = fullfile(fileparts(mfilename('fullpath')), 'output');
if ~exist(S.outDir, 'dir')
    mkdir(S.outDir);
end

% LiDAR -> camera extrinsics (adopted results)
S.R_lidar2flir = [ 0.048992   -0.998798   -0.00174722;
                    0.0621242   0.00479317 -0.998057;
                    0.996865    0.0487882   0.0622843 ];
S.t_lidar2flir = [-0.107859; -0.0426556; -0.0135286];

S.R_lidar2zed = [ 0.00758424 -0.999954   -0.00584239;
                  -0.0387419   0.00554434 -0.999234;
                   0.99922     0.00780478 -0.0386981 ];
S.t_lidar2zed = [-0.0992363; 0.0888694; -0.000231952];

% Intrinsics, no-skew model
S.Kf = [570.4796        0  149.1501;
               0  545.4275  117.0047;
               0         0         1];
S.kFlir = [-0.4241, -0.1241];
S.pFlir = [-0.0053,  0.0025];
S.flirW = 336; S.flirH = 256;

S.Kz = [1412.3362         0  1012.8503;
               0  1414.4716   569.7181;
               0         0          1];
S.kZed = [-1.558253e-01, 9.026829e-03];
S.pZed = [ 6.208599e-04, 5.667587e-04];
S.zedW = 1920; S.zedH = 1080;

%% Sync manifest + bag (opened once, reused on every pose change)

fprintf('Loading sync manifest: %s\n', syncManifestPath);
manifest = jsondecode(fileread(syncManifestPath));
S.triplets = manifest.triplets;
S.nTriplets = numel(S.triplets);

if startIdx < 0 || startIdx >= S.nTriplets
    error('startIdx=%d out of range [0, %d]', startIdx, S.nTriplets-1);
end

fprintf('Opening bag: %s\n', bagPath);
S.bag = ros2bagreader(bagPath);

fprintf(['\nKeys: right arrow/n = next pose, left arrow/p = previous, ' ...
    's = save PNG, q/close window = quit.\n\n']);

%% Figure + first render

S.idx = startIdx;
S.imgHandle = [];
S.scatterHandle = [];
S.axHandle = [];

fig = figure('Name', 'FLIR on ZED - session 9 (viewer)');
set(fig, 'KeyPressFcn', @keyHandler);
guidata(fig, S);

renderFrame(fig);

end

%% --- Keyboard callback ---

function keyHandler(src, event)
    S = guidata(src);
    switch event.Key
        case {'rightarrow', 'n'}
            newIdx = min(S.idx + 1, S.nTriplets - 1);
        case {'leftarrow', 'p'}
            newIdx = max(S.idx - 1, 0);
        case 's'
            saveCurrentFrame(S);
            return
        case 'q'
            close(src);
            return
        otherwise
            return
    end
    if newIdx ~= S.idx
        S.idx = newIdx;
        guidata(src, S);
        renderFrame(src);
    else
        fprintf('Already at the limit (pose %d).\n', S.idx);
    end
end

%% --- Render one pose ---

function renderFrame(figHandle)
    S = guidata(figHandle);
    tr = S.triplets(S.idx + 1);

    % --- LiDAR scans in the accumulation window around the triplet ---
    targetT = tr.lidar.timestamp_lidar;
    sel = select(S.bag, 'Time', [targetT - S.lidarAccumHalfWindow_s, targetT + S.lidarAccumHalfWindow_s], ...
        'Topic', '/cloud_registered');
    if sel.NumMessages == 0
        warning('No /cloud_registered within +-%.2fs for pose %d, skipping.', S.lidarAccumHalfWindow_s, S.idx);
        return
    end
    msgs = readMessages(sel);
    ptsWorld = cell(numel(msgs), 1);
    for i = 1:numel(msgs)
        ptsWorld{i} = rosReadXYZ(msgs{i});
    end
    ptsWorld = vertcat(ptsWorld{:});

    % --- world -> body using the /Odometry pose of the triplet ---
    t_wb = tr.lidar.position(:);
    q_xyzw = tr.lidar.orientation(:)';
    q_wxyz = [q_xyzw(4), q_xyzw(1), q_xyzw(2), q_xyzw(3)];
    R_wb = quat2rotm(q_wxyz);
    ptsBody = (R_wb' * (ptsWorld' - t_wb))';

    % --- projection into FLIR and ZED ---
    ptsFlir = (S.R_lidar2flir * ptsBody' + S.t_lidar2flir)';
    ptsZed  = (S.R_lidar2zed  * ptsBody' + S.t_lidar2zed)';

    [uFlir, vFlir, validFlir] = projectPinhole(ptsFlir, S.Kf, S.kFlir, S.pFlir, S.flirW, S.flirH);
    [uZed,  vZed,  validZed ] = projectPinhole(ptsZed,  S.Kz, S.kZed,  S.pZed,  S.zedW,  S.zedH);
    validBoth = validFlir & validZed;

    % per-camera z-buffer: drop the occluded points (not the nearest one in
    % their pixel), computed only on the subset already valid in both
    okFlir = false(size(validBoth)); okZed = false(size(validBoth));
    okFlir(validBoth) = zBufferMask(uFlir(validBoth), vFlir(validBoth), ptsFlir(validBoth,3), ...
        S.flirW, S.flirH, S.zBufferTol_m);
    okZed(validBoth)  = zBufferMask(uZed(validBoth),  vZed(validBoth),  ptsZed(validBoth,3), ...
        S.zedW,  S.zedH,  S.zBufferTol_m);
    validBoth = validBoth & okFlir & okZed;

    % --- colourised FLIR image ---
    [~, flirBase, ~] = fileparts(tr.flir.file);
    flirBase = erase(flirBase, '_R');
    flirNpyPath = fullfile(S.flirRot180Dir, [flirBase '.npy']);
    flirRaw = readNpyFloat32(flirNpyPath);
    flirGray = mat2gray(flirRaw);
    cmap = hot(256);
    flirIdx = min(max(round(flirGray * 255) + 1, 1), 256);
    flirRgb = ind2rgb(flirIdx, cmap);

    uF = round(uFlir(validBoth)); vF = round(vFlir(validBoth));
    uZ = uZed(validBoth);         vZ = vZed(validBoth);
    linIdx = sub2ind([S.flirH, S.flirW], vF, uF);
    rCh = flirRgb(:,:,1); gCh = flirRgb(:,:,2); bCh = flirRgb(:,:,3);
    sampledColors = [rCh(linIdx), gCh(linIdx), bCh(linIdx)];

    % --- ZED image ---
    zedImg = imread(fullfile(S.zedSessionDir, 'frames', tr.zed.file));

    % --- draw (reuse the handles if they already exist, much faster) ---
    if isempty(S.imgHandle) || ~isvalid(S.imgHandle)
        clf(figHandle);
        S.axHandle = axes('Parent', figHandle);
        S.imgHandle = imshow(zedImg, 'Parent', S.axHandle);
        hold(S.axHandle, 'on');
        S.scatterHandle = scatter(S.axHandle, uZ, vZ, 12, sampledColors, 'filled', 'MarkerFaceAlpha', 0.75);
    else
        set(S.imgHandle, 'CData', zedImg);
        set(S.scatterHandle, 'XData', uZ, 'YData', vZ, 'CData', sampledColors);
    end
    title(S.axHandle, sprintf('Session 9, pose %d/%d (%s) — FLIR %s on ZED %s', ...
        S.idx, S.nTriplets - 1, tr.match_status, tr.flir.file, tr.zed.file), 'Interpreter', 'none');
    drawnow;

    fprintf('Pose %d/%d | match=%-13s | FLIR %s | ZED %s | scans fused=%d | valid points=%d\n', ...
        S.idx, S.nTriplets - 1, tr.match_status, tr.flir.file, tr.zed.file, numel(msgs), sum(validBoth));

    guidata(figHandle, S);
end

%% --- Manual save ('s' key) ---

function saveCurrentFrame(S)
    outPng = fullfile(S.outDir, sprintf('flir_on_zed_session9_pose%02d.png', S.idx));
    exportgraphics(S.axHandle, outPng, 'Resolution', 200);
    fprintf('Saved: %s\n', outPng);
end

%% --- Projection / .npy reading functions ---

function mask = zBufferMask(u, v, z, W, H, tol)
% For each integer pixel (round(u),round(v)), keeps only the points within "tol"
% of the minimum depth observed in that pixel; discards the others (occluded by
% something nearer along the same camera ray).
    if isempty(u)
        mask = false(0,1);
        return
    end
    uu = min(max(round(u), 1), W);
    vv = min(max(round(v), 1), H);
    binIdx = sub2ind([H, W], vv, uu);
    z = double(z);
    minZ = accumarray(binIdx, z, [W*H, 1], @min, Inf);
    mask = z <= minZ(binIdx) + tol;
end

function [u, v, valid] = projectPinhole(P, K, k, p, W, H)
    z = P(:,3);
    valid = z > 0.05;
    xn = P(:,1) ./ z;
    yn = P(:,2) ./ z;
    r2 = xn.^2 + yn.^2;
    radial = 1 + k(1)*r2 + k(2)*r2.^2;
    xd = xn .* radial + 2*p(1)*xn.*yn + p(2)*(r2 + 2*xn.^2);
    yd = yn .* radial + p(1)*(r2 + 2*yn.^2) + 2*p(2)*xn.*yn;
    u = K(1,1)*xd + K(1,3);
    v = K(2,2)*yd + K(2,3);
    valid = valid & u >= 1 & u <= W & v >= 1 & v <= H;
end

function arr = readNpyFloat32(npyPath)
    fid = fopen(npyPath, 'r');
    if fid < 0
        error('Cannot open %s', npyPath);
    end
    cleanupObj = onCleanup(@() fclose(fid));
    fread(fid, 6, 'uint8=>char');
    fread(fid, 2, 'uint8');
    headerLen = fread(fid, 1, 'uint16');
    headerStr = fread(fid, headerLen, 'uint8=>char')';

    shapeTok = regexp(headerStr, "'shape':\s*\(([^)]*)\)", 'tokens', 'once');
    dims = str2double(strsplit(strtrim(shapeTok{1}), ','));
    dims(isnan(dims)) = [];
    nRows = dims(1); nCols = dims(2);

    if ~contains(headerStr, '<f4')
        error('Unhandled .npy format (expected float32 little-endian ''<f4''): %s', headerStr);
    end
    data = fread(fid, nRows * nCols, 'single=>single');
    arr = reshape(data, [nCols, nRows])';
end
