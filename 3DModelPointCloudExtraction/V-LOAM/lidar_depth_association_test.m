%% lidar_depth_association_test.m
% V-LOAM-style (Zhang & Singh, ICRA 2015) depth association test on
% Session 7: give ORB keypoints in ZED frames REAL metric depth from the
% LiDAR, instead of letting monocular VO (monovslam_test.m) carry an
% arbitrary scale.
%
% This is a VERIFICATION step only (task: "verificare se l'associazione di
% profondita' funziona bene su Session 7, dato il rig grazing-angle
% laterale del rover, come primo passo prima di usarla per dare scala
% metrica a monovslam"). It does NOT estimate a trajectory.
%
% Simplification vs. the paper (explicitly allowed by the task): the paper
% (Sec. V) keeps a depthmap as a 2D KD-tree in spherical coordinates, finds
% the 3 closest depthmap points per visual feature, forms a local planar
% patch, and gets the distance by ray/plane intersection. Here we instead
% do a single nearest-projected-LiDAR-point-within-a-pixel-radius lookup
% per ORB keypoint. Good enough to answer "does depth coverage exist here
% at all", not meant to be the final depth estimator.
%
% Pipeline:
%   1) lidar_zed_depth_sync.py (Python helper, same folder) does the parts
%      that need `rosbags` (no MATLAB path exists for this custom-message
%      rosbag2 in this project) and the ZED<->LiDAR time sync -- REUSING,
%      not reimplementing, sync_manifest.py's load_zed_frames/nearest_index
%      and verify_loops_appearance.py's read_poses_tum_with_ts (see that
%      script's docstring for the full reuse map and the LiDAR-point-source
%      decision, made with the user, not guessed). Its output (this
%      script's input) is depth_assoc_out/{rig_calibration_resolved.json,
%      lidar_depth_assoc_manifest.csv, lidar_depth_assoc_points.csv}.
%   2) THIS script projects each sampled keyframe's LiDAR points (already
%      in the LiDAR/laser sensor frame) into the ZED image with
%      rig_calibration.yaml's T_lidar_to_zed extrinsics + the ZED K/dist
%      rescaled to 1920x1080 (rig_calibration.py's zed_K_for), extracts ORB
%      keypoints, and associates depth by nearest-pixel lookup.
%   3) Saves a per-keyframe coverage CSV and overlay PNGs for the
%      worst/median/best coverage keyframes found.
%
% IMPORTANT CAVEATS (do not silently trust these, see printed summary):
%   - --lidar-zed-offset is UNVERIFIED (assumed 0, shared host clock) --
%     see lidar_zed_depth_sync.py's docstring and its printed warning.
%   - poses_tum_matlab.txt (the "already elaborated" pose-graph/loop-closed
%     LiDAR trajectory) is only 137 rows over ~397s of Session 7 -- very
%     sparse (uneven spacing, several tens of seconds between some rows).
%     Using it as "the nearest LiDAR pose" per the task's step 1 is
%     reported (posetum_delta_s column) but is a POOR match for many
%     keyframes. The actual 3D points used here come from the raw
%     /cloud_registered scan nearest each ZED frame instead (cloud_delta_s
%     column, sub-0.1s for every sampled keyframe in this run) -- see
%     lidar_zed_depth_sync.py's docstring for why.
%   - /Odometry (FAST-LIO) pose used to invert /cloud_registered back to
%     the LiDAR/laser frame is the estimator's body/IMU frame, not
%     necessarily the LiDAR's own optical origin (assumed negligible for a
%     Livox integrated unit, same implicit assumption as the rest of the
%     project's LiDAR<->camera extrinsics). Not separately corrected.
%
% Non modifica alcun file esistente del progetto o della sessione dati.

clear; clc; close all;

%% --- Config ---
scriptDir = fileparts(mfilename('fullpath'));
outDir    = fullfile(scriptDir, 'depth_assoc_out');
pyExe     = 'C:\venvs\sensorfusion\Scripts\python.exe';
pyHelper  = fullfile(scriptDir, 'lidar_zed_depth_sync.py');

zedSession = 'C:\Users\loren\Desktop\Dati_vfinal\SLAM\ZED\20260730_155047';
zedFramesDir = fullfile(zedSession, 'frames');

calibJson    = fullfile(outDir, 'rig_calibration_resolved.json');
manifestCsv  = fullfile(outDir, 'lidar_depth_assoc_manifest.csv');
pointsCsv    = fullfile(outDir, 'lidar_depth_assoc_points.csv');

pxAssocThreshold = 3.0;   % pixel radius for ORB<->projected-LiDAR-point association
nExamplePlots    = 3;     % worst / median / best coverage keyframes get an overlay PNG

%% --- Stage 1: run the Python sync/extraction helper if its outputs are missing ---
needSync = ~(isfile(calibJson) && isfile(manifestCsv) && isfile(pointsCsv));
if needSync
    if ~isfile(pyExe)
        error(['lidar_depth_association_test:missingPython', newline, ...
            'venv Python non trovato: %s', newline, ...
            'Il pipeline di sync ZED<->LiDAR (rosbags) gira solo li'' (vedi ', ...
            'lidar_zed_depth_sync.py). Crea/aggiusta il venv "sensorfusion" o ', ...
            'esegui lo script Python a mano e rilancia questo .m.'], pyExe);
    end
    cmd = sprintf('"%s" "%s"', pyExe, pyHelper);
    fprintf('Output di sync mancanti in %s -- eseguo:\n  %s\n', outDir, cmd);
    status = system(cmd);
    if status ~= 0 || ~(isfile(calibJson) && isfile(manifestCsv) && isfile(pointsCsv))
        error(['lidar_depth_association_test:syncFailed', newline, ...
            'lidar_zed_depth_sync.py e'' terminato con status %d o non ha scritto ', ...
            'tutti gli output attesi in %s. Esegui a mano per vedere l''errore:', ...
            newline, '  %s'], status, outDir, cmd);
    end
end

%% --- Load calibration (resolved by rig_calibration.py, NOT re-derived here) ---
calib = jsondecode(fileread(calibJson));
fprintf('Calibrazione: %s\n', calib.source_rig_calibration);
K = calib.zed_K_1920x1080;
d = calib.zed_dist;
imageSize = [calib.zed_image_size.height, calib.zed_image_size.width];
intrinsics = cameraIntrinsics([K.fx K.fy], [K.cx K.cy], imageSize, ...
    RadialDistortion=[d.k1 d.k2 d.k3], TangentialDistortion=[d.p1 d.p2]);
T_lidar_to_zed = calib.T_lidar_to_zed;   % 4x4, laser frame -> ZED camera frame
identityTform = rigidtform3d(eye(3), [0 0 0]);  % points already in camera frame

%% --- Load sync manifest + LiDAR points ---
manifest = readtable(manifestCsv);
pointsAll = readtable(pointsCsv);
nSamples = height(manifest);
fprintf('Manifest: %d keyframe(s) campionati, %d punto/i LiDAR totali (cap %d/scan lato Python)\n', ...
    nSamples, height(pointsAll), max(groupcounts(pointsAll.cloud_ordinal)));

fprintf('Sync: posetum_delta_s (pose "gia'' elaborata" piu'' vicina) min/median/max = %.2f / %.2f / %.2f s\n', ...
    min(manifest.posetum_delta_s), median(manifest.posetum_delta_s), max(manifest.posetum_delta_s));
fprintf('Sync: cloud_delta_s (scan raw /cloud_registered usato per i punti) min/median/max = %.3f / %.3f / %.3f s\n', ...
    min(abs(manifest.cloud_delta_s)), median(abs(manifest.cloud_delta_s)), max(abs(manifest.cloud_delta_s)));
if max(abs(manifest.posetum_delta_s)) > 5
    warning('lidar_depth_association_test:sparsePosetum', ...
        ['poses_tum_matlab.txt e'' troppo sparso per essere usato come pose LiDAR per-frame ', ...
         '(delta fino a %.1fs): i punti 3D qui vengono dallo scan raw /cloud_registered piu'' ', ...
         'vicino, NON da poses_tum_matlab.txt (vedi commento in testa allo script).'], ...
        max(abs(manifest.posetum_delta_s)));
end

%% --- Per-keyframe: project LiDAR, detect ORB, associate depth ---
summary = table('Size', [nSamples 6], ...
    'VariableTypes', {'double','string','double','double','double','double'}, ...
    'VariableNames', {'sample_idx','zed_file','n_orb_total','n_with_depth','n_without_depth','coverage_pct'});

perSample = struct('sample_idx', {}, 'zedPath', {}, 'orbXY', {}, 'hasDepth', {}, ...
    'depthVal', {}, 'lidarUV', {}, 'lidarDepth', {});

for i = 1:nSamples
    row = manifest(i, :);
    zedPath = fullfile(zedFramesDir, row.zed_file{1});
    if ~isfile(zedPath)
        warning('ZED frame mancante, salto keyframe %d: %s', row.sample_idx, zedPath);
        continue;
    end

    % --- LiDAR points for this keyframe's matched scan, laser frame -> camera frame ---
    mask = pointsAll.cloud_ordinal == row.cloud_ordinal;
    pLidar = [pointsAll.x_lidar(mask), pointsAll.y_lidar(mask), pointsAll.z_lidar(mask)];
    pCamH = [pLidar, ones(size(pLidar,1),1)] * T_lidar_to_zed';   % (N x 4) * (4 x 4)'
    pCam = pCamH(:, 1:3);
    inFront = pCam(:,3) > 0;
    pCam = pCam(inFront, :);

    [lidarUV, valid] = world2img(pCam, identityTform, intrinsics, ApplyDistortion=true);
    lidarUV = lidarUV(valid, :);
    lidarDepth = pCam(valid, 3);
    inBounds = lidarUV(:,1) >= 1 & lidarUV(:,1) <= imageSize(2) & ...
               lidarUV(:,2) >= 1 & lidarUV(:,2) <= imageSize(1);
    lidarUV = lidarUV(inBounds, :);
    lidarDepth = lidarDepth(inBounds);

    % --- ORB keypoints ---
    I = imread(zedPath);
    if size(I,3) == 3
        Igray = rgb2gray(I);
    else
        Igray = I;
    end
    orbPts = detectORBFeatures(Igray);
    orbXY = orbPts.Location;   % N x 2, [x y] same convention as world2img output
    nOrb = size(orbXY, 1);

    % --- nearest-neighbor association within pxAssocThreshold (brute-force,
    %     no KD-tree / Statistics Toolbox dependency -- see header comment) ---
    hasDepth = false(nOrb, 1);
    depthVal = nan(nOrb, 1);
    if nOrb > 0 && ~isempty(lidarUV)
        dx = orbXY(:,1) - lidarUV(:,1)';   % nOrb x nLidar, implicit expansion
        dy = orbXY(:,2) - lidarUV(:,2)';
        D2 = dx.^2 + dy.^2;
        [minD2, minIdx] = min(D2, [], 2);
        hasDepth = minD2 <= pxAssocThreshold^2;
        depthVal(hasDepth) = lidarDepth(minIdx(hasDepth));
    end

    nWith = sum(hasDepth);
    summary.sample_idx(i) = row.sample_idx;
    summary.zed_file(i) = row.zed_file{1};
    summary.n_orb_total(i) = nOrb;
    summary.n_with_depth(i) = nWith;
    summary.n_without_depth(i) = nOrb - nWith;
    summary.coverage_pct(i) = 100 * nWith / max(nOrb, 1);

    perSample(i).sample_idx = row.sample_idx;
    perSample(i).zedPath = zedPath;
    perSample(i).orbXY = orbXY;
    perSample(i).hasDepth = hasDepth;
    perSample(i).depthVal = depthVal;
    perSample(i).lidarUV = lidarUV;
    perSample(i).lidarDepth = lidarDepth;

    if mod(i, 20) == 0
        fprintf('  keyframe %d/%d: %d ORB, %d con profondita'' (%.1f%%)\n', ...
            i, nSamples, nOrb, nWith, summary.coverage_pct(i));
    end
end

%% --- Save summary CSV ---
summaryOut = [manifest(:, {'sample_idx','zed_file','posetum_delta_s','cloud_delta_s'}), ...
    summary(:, {'n_orb_total','n_with_depth','n_without_depth','coverage_pct'})];
summaryCsvPath = fullfile(outDir, 'depth_assoc_summary.csv');
writetable(summaryOut, summaryCsvPath);
fprintf('\nSalvato: %s\n', summaryCsvPath);

fprintf('\n=== Copertura profondita'' (soglia associazione %.1f px) ===\n', pxAssocThreshold);
fprintf('  Keyframe validi: %d/%d\n', sum(summary.n_orb_total > 0), nSamples);
fprintf('  Coverage %% -- min/median/mean/max: %.1f / %.1f / %.1f / %.1f\n', ...
    min(summary.coverage_pct), median(summary.coverage_pct), ...
    mean(summary.coverage_pct), max(summary.coverage_pct));
fprintf(['  NOTA: rig con LiDAR grazing-angle laterale sul rover -- ci si aspetta copertura ', ...
    'bassa/irregolare specialmente sui bordi immagine; questo test serve a QUANTIFICARLO, ', ...
    'non a correggerlo.\n']);

%% --- Overlay plots: worst / median / best coverage keyframe ---
validRows = find(summary.n_orb_total > 0);
[~, order] = sort(summary.coverage_pct(validRows));
pickIdx = validRows(order([1, max(1,round(numel(order)/2)), numel(order)]));
pickTag = {'worst', 'median', 'best'};

for k = 1:min(nExamplePlots, numel(pickIdx))
    i = pickIdx(k);
    s = perSample(i);
    I = imread(s.zedPath);

    fig = figure('Visible', 'off', 'Color', 'w', 'Position', [100 100 1200 700]);
    imshow(I); hold on;
    hLidar = scatter(s.lidarUV(:,1), s.lidarUV(:,2), 4, s.lidarDepth, 'filled');
    colormap(gca, 'jet'); cb = colorbar; cb.Label.String = 'profondita'' LiDAR (m)';
    withD = s.hasDepth; withoutD = ~s.hasDepth;
    hWith = plot(s.orbXY(withD,1), s.orbXY(withD,2), 'go', 'MarkerSize', 8, 'LineWidth', 1.2);
    hWithout = plot(s.orbXY(withoutD,1), s.orbXY(withoutD,2), 'ro', 'MarkerSize', 8, 'LineWidth', 1.2);
    title(sprintf('%s (idx %d, %s): ORB %d, con profondita'' %d (%.1f%%)  [assoc <= %.1f px]', ...
        pickTag{k}, s.sample_idx, strrep(summary.zed_file{i}, '_', '\_'), ...
        summary.n_orb_total(i), summary.n_with_depth(i), summary.coverage_pct(i), pxAssocThreshold), ...
        'Interpreter', 'tex');
    % plot() on a fully-empty selection (0% or 100% coverage keyframes, e.g.
    % worst case here) returns a 0x0 handle that silently drops out of a
    % concatenated handle array -- pairing legend labels positionally
    % against [hLidar hWith hWithout] would then mislabel the remaining
    % series. Build handles/labels together so they always stay paired.
    legHandles = hLidar;
    legLabels = {'punti LiDAR proiettati'};
    if any(withD)
        legHandles = [legHandles hWith]; %#ok<AGROW>
        legLabels{end+1} = 'ORB con profondita''';
    end
    if any(withoutD)
        legHandles = [legHandles hWithout]; %#ok<AGROW>
        legLabels{end+1} = 'ORB senza profondita''';
    end
    legend(legHandles, legLabels, 'Location', 'southoutside', 'Orientation', 'horizontal');

    pngPath = fullfile(outDir, sprintf('overlay_%s_idx%03d.png', pickTag{k}, s.sample_idx));
    exportgraphics(fig, pngPath, 'Resolution', 150);
    close(fig);
    fprintf('Salvato overlay (%s): %s\n', pickTag{k}, pngPath);
end

fprintf('\nFATTO. Vedi %s per CSV/PNG. Nessun file esistente del progetto/dei dati e'' stato modificato.\n', outDir);
