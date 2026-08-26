function FlirLidarZedViewer(startIdx)
%FLIRLIDARZEDVIEWER Overlay interattivo FLIR->ZED, sessione 9, scorribile pose per pose.
%
% Per i dettagli e le fonti dei parametri (intrinseche, estrinseche, convenzione
% FLIR rot180) vedi README.md in questa cartella. Navigazione: ogni pressione di tasto
% carica il TRIPLET successivo/precedente dal sync manifest -> nuova scansione
% LiDAR dal bag, nuova immagine FLIR, nuovo frame ZED, nuova proiezione.
%
% Uso:
%   FlirLidarZedViewer        % parte dal pose 9
%   FlirLidarZedViewer(30)    % parte dal pose 30
%
% Tasti (finestra della figura deve avere il focus):
%   freccia destra / n   -> pose successivo
%   freccia sinistra / p -> pose precedente
%   s                     -> salva il frame corrente in output/
%   q / chiusura finestra -> esce
%
% NOTA: richiede una sessione MATLAB interattiva (desktop). Non funziona con
% `matlab -batch`, perche' in batch la finestra si chiude appena lo script
% termina e non resta nessun event loop ad ascoltare i tasti.

close all
clc
% NB: niente "clear" qui - in una function cancellerebbe anche l'argomento
% di ingresso startIdx prima ancora di leggerlo (ogni chiamata a una
% function parte comunque con workspace pulito, non serve).

if nargin < 1
    startIdx = 9;
end

%% Parametri fissi (fonti documentate in README.md)

sessionRoot   = 'C:\Users\loren\Desktop\Dati_vfinal\SLAM';
S.zedSessionDir = fullfile(sessionRoot, 'ZED', '20260730_161223', 'fullrate');
S.flirRot180Dir = fullfile(sessionRoot, 'Flir', 'session9_only_rot180');
bagPath       = fullfile(sessionRoot, 'Lidar', 'rosbag2_2026_07_30-18_12_20');
syncManifestPath = fullfile(S.zedSessionDir, 'sync_manifest.json');

S.lidarSearchWindow_s = 1.0;

% Il Livox HAP ha scansione non ripetitiva: UNA sola scansione (~0.2s, lo
% spacing tra messaggi /cloud_registered) copre solo bande parziali della
% scena, da cui le righe larghe visibili sull'overlay. Per infittire, si
% fondono piu' scansioni consecutive attorno al pose scelto (tutte
% trasformate world->body con la STESSA posa /Odometry del triplet, quindi
% assunzione di robot quasi fermo durante la finestra: piu' larga = piu'
% punti ma piu' "motion blur" per superfici in movimento nella scena).
S.lidarAccumHalfWindow_s = 0.4;   % secondi PRIMA e DOPO il timestamp del pose

% Filtro di occlusione (z-buffer) per camera: FLIR e ZED non sono co-locate
% (baseline reale ~13cm tra loro sul rig), quindi su uno spigolo vicino le due
% camere vedono "dietro l'angolo" in modo diverso. Senza questo filtro, un
% punto della parete nascosta dietro uno spigolo (invisibile a FLIR ma
% geometricamente dentro il suo frustum) viene comunque campionato,
% prendendo per errore il colore dello spigolo in primo piano. Per ogni
% pixel (int) di ciascuna camera si tiene solo il punto piu' vicino (± tol).
S.zBufferTol_m = 0.08;   % margine oltre il piu' vicino, stesso ordine dell'RMSE di calibrazione
S.outDir = fullfile(fileparts(mfilename('fullpath')), 'output');
if ~exist(S.outDir, 'dir')
    mkdir(S.outDir);
end

% Estrinseche LiDAR -> camera (risultati adottati)
S.R_lidar2flir = [ 0.048992   -0.998798   -0.00174722;
                    0.0621242   0.00479317 -0.998057;
                    0.996865    0.0487882   0.0622843 ];
S.t_lidar2flir = [-0.107859; -0.0426556; -0.0135286];

S.R_lidar2zed = [ 0.00758424 -0.999954   -0.00584239;
                  -0.0387419   0.00554434 -0.999234;
                   0.99922     0.00780478 -0.0386981 ];
S.t_lidar2zed = [-0.0992363; 0.0888694; -0.000231952];

% Intrinseche, modello senza skew
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

%% Sync manifest + bag (aperti una volta sola, riusati ad ogni cambio pose)

fprintf('Carico sync manifest: %s\n', syncManifestPath);
manifest = jsondecode(fileread(syncManifestPath));
S.triplets = manifest.triplets;
S.nTriplets = numel(S.triplets);

if startIdx < 0 || startIdx >= S.nTriplets
    error('startIdx=%d fuori range [0, %d]', startIdx, S.nTriplets-1);
end

fprintf('Apro bag: %s\n', bagPath);
S.bag = ros2bagreader(bagPath);

fprintf(['\nTasti: freccia destra/n = pose successivo, freccia sinistra/p = precedente, ' ...
    's = salva PNG, q/chiudi finestra = esci.\n\n']);

%% Figura + primo render

S.idx = startIdx;
S.imgHandle = [];
S.scatterHandle = [];
S.axHandle = [];

fig = figure('Name', 'FLIR su ZED - sessione 9 (viewer)');
set(fig, 'KeyPressFcn', @keyHandler);
guidata(fig, S);

renderFrame(fig);

end

%% --- Callback tastiera ---

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
        fprintf('Gia'' al limite (pose %d).\n', S.idx);
    end
end

%% --- Render di un pose ---

function renderFrame(figHandle)
    S = guidata(figHandle);
    tr = S.triplets(S.idx + 1);

    % --- scansioni LiDAR nella finestra di accumulo attorno al triplet ---
    targetT = tr.lidar.timestamp_lidar;
    sel = select(S.bag, 'Time', [targetT - S.lidarAccumHalfWindow_s, targetT + S.lidarAccumHalfWindow_s], ...
        'Topic', '/cloud_registered');
    if sel.NumMessages == 0
        warning('Nessun /cloud_registered entro +-%.2fs per pose %d, skip.', S.lidarAccumHalfWindow_s, S.idx);
        return
    end
    msgs = readMessages(sel);
    ptsWorld = cell(numel(msgs), 1);
    for i = 1:numel(msgs)
        ptsWorld{i} = rosReadXYZ(msgs{i});
    end
    ptsWorld = vertcat(ptsWorld{:});

    % --- world -> body con la posa /Odometry del triplet ---
    t_wb = tr.lidar.position(:);
    q_xyzw = tr.lidar.orientation(:)';
    q_wxyz = [q_xyzw(4), q_xyzw(1), q_xyzw(2), q_xyzw(3)];
    R_wb = quat2rotm(q_wxyz);
    ptsBody = (R_wb' * (ptsWorld' - t_wb))';

    % --- proiezione in FLIR e ZED ---
    ptsFlir = (S.R_lidar2flir * ptsBody' + S.t_lidar2flir)';
    ptsZed  = (S.R_lidar2zed  * ptsBody' + S.t_lidar2zed)';

    [uFlir, vFlir, validFlir] = projectPinhole(ptsFlir, S.Kf, S.kFlir, S.pFlir, S.flirW, S.flirH);
    [uZed,  vZed,  validZed ] = projectPinhole(ptsZed,  S.Kz, S.kZed,  S.pZed,  S.zedW,  S.zedH);
    validBoth = validFlir & validZed;

    % z-buffer per camera: scarta i punti occlusi (non i piu' vicini nel loro
    % pixel), calcolato solo sul sottoinsieme gia' valido in entrambe
    okFlir = false(size(validBoth)); okZed = false(size(validBoth));
    okFlir(validBoth) = zBufferMask(uFlir(validBoth), vFlir(validBoth), ptsFlir(validBoth,3), ...
        S.flirW, S.flirH, S.zBufferTol_m);
    okZed(validBoth)  = zBufferMask(uZed(validBoth),  vZed(validBoth),  ptsZed(validBoth,3), ...
        S.zedW,  S.zedH,  S.zBufferTol_m);
    validBoth = validBoth & okFlir & okZed;

    % --- immagine FLIR colorizzata ---
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

    % --- immagine ZED ---
    zedImg = imread(fullfile(S.zedSessionDir, 'frames', tr.zed.file));

    % --- disegna (riusa gli handle se gia' esistono, molto piu' veloce) ---
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
    title(S.axHandle, sprintf('Sessione 9, pose %d/%d (%s) — FLIR %s su ZED %s', ...
        S.idx, S.nTriplets - 1, tr.match_status, tr.flir.file, tr.zed.file), 'Interpreter', 'none');
    drawnow;

    fprintf('Pose %d/%d | match=%-13s | FLIR %s | ZED %s | scan fuse=%d | punti validi=%d\n', ...
        S.idx, S.nTriplets - 1, tr.match_status, tr.flir.file, tr.zed.file, numel(msgs), sum(validBoth));

    guidata(figHandle, S);
end

%% --- Salvataggio manuale (tasto 's') ---

function saveCurrentFrame(S)
    outPng = fullfile(S.outDir, sprintf('flir_on_zed_session9_pose%02d.png', S.idx));
    exportgraphics(S.axHandle, outPng, 'Resolution', 200);
    fprintf('Salvato: %s\n', outPng);
end

%% --- Funzioni di proiezione / lettura .npy ---

function mask = zBufferMask(u, v, z, W, H, tol)
% Per ogni pixel intero (round(u),round(v)), tiene solo i punti entro "tol"
% dalla profondita' minima osservata in quel pixel; scarta gli altri
% (occlusi da qualcosa di piu' vicino lungo lo stesso raggio della camera).
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
        error('Impossibile aprire %s', npyPath);
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
        error('Formato .npy non gestito (atteso float32 little-endian ''<f4''): %s', headerStr);
    end
    data = fread(fid, nRows * nCols, 'single=>single');
    arr = reshape(data, [nCols, nRows])';
end
