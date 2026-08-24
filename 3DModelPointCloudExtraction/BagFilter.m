%% Loop closure on an already-processed FAST-LIO bag
%
% PROBLEM IT SOLVES
% /cloud_registered is the output of FAST-LIO, so the points are already in
% the map frame BUT with the drift baked into the transforms. The typical
% symptom is a flat corridor that "rises" several meters in the map.
%
% APPROACH
% 1. Poses are read from /Odometry
% 2. The clouds are brought back to the body frame by inverting the pose (un-transform)
% 3. Keyframes are selected
% 4. Loops are searched with Scan Context
% 5. Loops are verified with ICP, discarding false positives
% 6. A pose graph is built with sequential + loop constraints
% 7. It is optimized and the map is reconstructed with the corrected poses
%
% REQUIREMENTS: ROS Toolbox, Lidar Toolbox, Navigation Toolbox, Computer Vision Toolbox

clear
close all
clc

%% 1. Parameters
bagPath = "C:\Users\loren\Desktop\Dati_vfinal\SLAM\Lidar\rosbag2_2026_07_30-18_12_20\rosbag2_2026_07_30-18_12_20_0.db3";

% Keyframe selection: a new keyframe when the sensor has moved by
% kfDistance meters OR rotated by kfAngle degrees relative to the previous one.
kfDistance = 1.0;    % m
kfAngle    = 15;     % degrees

% Cut the trajectory to a [start, end] window in seconds (section 5b).
% Useful both to discard the tail of the path (e.g. the return leg that
% overlaps the outbound corridor, creating duplicates in the map) and to
% isolate a segment and try loop closure only there, without guessing a
% spatial threshold: an X coordinate can be crossed multiple times
% unpredictably (turns, local back-and-forth), elapsed time cannot. Tune by
% looking at "Usable odometry duration" (Section 3b) and the "Trajectory"
% plot of a previous run with useTimeCut = false.
useTimeCut      = false;
timeCutStartSec = 10;   % s, discard keyframes with elapsed time < this (NaN = from the start)
timeCutEndSec   = 180;   % s, discard keyframes with elapsed time > this (NaN = to the end)

% Topological classification of keyframes (node/line), section 6d.
% Inspired by Xu et al., "When-to-Loop: Enhanced Loop Closure for LiDAR SLAM
% in Urban Environments Based on Scan Context" (Micromachines 2024):
% intersections/T-junctions are more reliable for loop closure (revisitable
% from multiple directions), straight stretches are the category at risk of
% false positives from repetitive structure (doors, evenly spaced columns).
% The paper's ground/obstacle segmentation is NOT done here: cumulative
% rotation is used as a proxy instead (dAng, as in Section 5, but between
% consecutive keyframes) over a window of adjacent keyframes, to capture
% turns spread over several steps and not just single jumps.
% nodeAngThreshold is deliberately higher than kfAngle: it must represent a
% real cumulative turn, not the noise that already triggers a new keyframe
% by itself.
nodeAngWindow    = 3;    % keyframe radius of the window (total 2w+1)
nodeAngThreshold = 45;   % degrees, cumulative threshold above which the keyframe is a "node"

% Downsampling applied to each keyframe, used both by Scan Context and ICP
kfVoxel    = 0.05;   % m

% Outlier filter (statistical, pcdenoise): for each point, look at the mean
% distance to its NumNeighbors nearest neighbors; points whose mean distance
% exceeds mean + Threshold*standard_deviation (computed over the whole
% cloud) are discarded. Applied at two points in the pipeline:
%   - per keyframe, right after extraction (Section 6, this also helps the
%     later fits: floor 6b, wall normals 6c, and the ICP correspondences of
%     Section 8);
%   - on the final reconstructed map (Section 11), where "seam" outliers
%     between different scans also appear, which do not exist in the single
%     keyframe.
useOutlierFilterKF    = true;
outlierKFNumNeighbors = 8;      % neighboring points used for the statistic
outlierKFStdFactor    = 2;    % threshold in standard deviations, lower = more aggressive

useOutlierFilterMap    = true;
outlierMapNumNeighbors = 12;
outlierMapStdFactor    = 2.0;

% Attitude realignment on the floor (see section 6b).
% Correction of roll/pitch drift using the floor as the gravity reference.
% Disable only if the environment does NOT have a flat floor (outdoors,
% uneven terrain, continuous ramps).
useGravityAlign = true;
floorBand    = 0.30;   % m, thickness of the low band in which to search for the floor
floorTol     = 0.06;   % m, planarity tolerance of the fit
floorMaxTilt = 50;     % degrees, max accepted tilt for the plane found

% YAW drift correction on the walls (section 6c).
% The floor constrains roll and pitch but NOT yaw: corridors remain
% non-perpendicular in plan. The walls are the reference for the third DOF.
% MEASURED on this dataset: wall azimuth 89.45 deg in the first half versus
% 82.22 in the second (a ~7 deg step at the same keyframe where roll/pitch
% broke). After the correction the two halves coincide and the dispersion
% drops from 0.082 to 0.014.
%
% WARNING: unlike the floor, this correction ASSUMES the building is
% orthogonal. Always check the printed dispersion: if it does not drop
% clearly, the assumption does not hold and it must be disabled.
useYawAlign = true;
wallMaxNz   = 0.2;   % |nz| below which a normal is considered a wall
yawRefKF    = 40;    % initial keyframes used as azimuth reference
yawSmooth   = 9;     % smoothing window of the estimate, in keyframes

% Gravity constraint INSIDE the pose graph (section 9). Without it, the
% optimization partly undoes the realignment from 6b in order to satisfy
% the loops.
% MEASURED on this dataset (Zdev / median tilt / mean node displacement):
%   6b without optimizing : 1.84 m  0.74 deg    -
%   loop without gravity   : 2.21 m  1.37 deg  0.21 m   <- the optimizer re-tilts
%   loop + gravity  50   : 1.82 m  0.66 deg  0.31 m
%   loop + gravity 5000   : 1.79 m  0.69 deg  2.19 m   <- not worth it
%
% The weight must NOT be raised arbitrarily: the gain on drift and tilt
% saturates around 50, while node displacement grows without bound. Going
% from 50 to 5000 buys 3 cm of vertical flatness and costs 1.9 m of XY
% displacement, which we have no way to validate.
useGravityFactor = true;
infoGravRP   = 50;     % weight on roll/pitch (comparable to infoOdom = 100)
infoGravFree = 1e-6;   % weight on the free DOFs (x,y,z,yaw): near-zero but > 0

% Z height constraint on the trajectory, from the same floor as 6b.
% Unlike the 13 loops (Section 9 below), floorDist is a FRESH LOCAL
% measurement at every keyframe (it does not carry over accumulated drift):
% that is why it corrects the residual Z drift that the loops, already
% agreeing with the odometry within 15 cm, have no room to correct (see note
% above at "Applying the loop constraints"). MEASURED on this dataset
% (variant AC.m, same floor): height std pre-optim. 0.44 m -> 0.15 m post,
% pose Z extent 1.82 m -> 0.69 m, with infoGravZ=5.
% floorFitMin: minimum fraction of keyframes with a valid floor fit for the
% constraint to activate; below that threshold the floor is not observed
% enough and only the roll/pitch constraint above is kept.
floorFitMin = 0.5;
infoGravZ   = 1e-6;

% Applying the loop constraints in the pose graph.
% MEASURED on this dataset (rosbag2_2026_07_30-17_50_45): the 13 valid loops
% agree with the odometry within 0.15 m and 3.4 deg, so they carry little
% corrective information. The effect is marginal and the two metrics do not
% agree:
%     realignment only : pose Z drift 1.84 m, map Z span 9.21 m
%     + loop closure      : pose Z drift 2.21 m, map Z span 8.87 m
% Here the true drift was in attitude (see 6b), not accumulation over
% revisits: it is the realignment doing the work, not the loops. On a
% dataset with real revisits and accumulated drift the loops matter much more.
useLoopClosure = true;

% Loop detection
scDistThreshold  = 0.15;   % Scan Context distance threshold, lower = more selective
scNumExcluded    = 30;     % recent keyframes excluded from the search
scMaxDetections  = 3;      % candidates per keyframe

% Loop search by spatial proximity (complementary to Scan Context)
proxRadius       = 3.0;    % m, XY radius within which two keyframes are "the same place"
proxMinGap       = 40;     % minimum keyframe separation to speak of a revisit
proxMaxCand      = 300;    % cap on proximity candidates, the closest ones

% ICP verification of loop candidates
icpMaxRMSE       = 0.30;   % m, above this threshold the loop is discarded
icpMaxDistance   = 1.0;    % m, maximum distance between correspondences

% Consistency filter with the odometry.
% A low RMSE is NOT enough to trust a loop: in corridors with repetitive
% structures (doors, evenly spaced columns) ICP can lock onto the wrong
% "repeat" and converge with an excellent RMSE but a completely wrong
% transform. The correct check is the 6-DOF comparison with the odometry
% estimate: loop closure is meant to correct a drift of tens of centimeters,
% not to flip the trajectory by meters. A constraint that contradicts the
% odometry beyond these thresholds is a false positive, not a correction.
%
% NB: filtering only the Z component is not enough. A wrong match can have a
% plausible Z and still be off by meters in XY and by degrees in rotation,
% distorting the entire graph once propagated.
loopMaxTransErr = 3.0;    % m, max translation discrepancy relative to odometry
loopMaxRotErr   = 10.0;   % degrees, max rotation discrepancy

% Relative weight of the constraints in the pose graph.
% WARNING: what matters is the TOTAL weight, not the weight per single
% constraint. With N keyframes there are N-1 odometry constraints but
% typically only a handful of loops: if infoLoop does not compensate for
% this numerical imbalance, the optimizer effectively ignores the loops even
% if they are correct, and the solution stays the initial odometric one (no
% visible correction). Here the weight of each loop is scaled by its actual
% ICP RMSE: a more precise match than sigma0 weighs more, one close to the
% threshold weighs less.
infoOdom = 100;
infoLoop = 500;    % reference weight for a loop with rmse = sigma0
sigma0Loop = kfVoxel;   % m, reference uncertainty (== voxel resolution)

% Additional weight for node/line topology (Section 6d), multiplicative on
% infoLoopEff below. MEASURED on this dataset: 12 of the 13 accepted loops
% involve at least one node, only 77->109 is line-line — nodes are a
% reliability signal for ACCEPTED loops, not a filter on false positives
% (72->133 was node-node and is still a false positive, already discarded
% by the odometry consistency check, not by this weight).
% topoWMixed = 1 leaves the behavior unchanged on node-line (the majority of
% the 13); node-node weighs more, line-line less.
topoWNodeNode = 1.5;
topoWMixed    = 1.0;
topoWLineLine = 0.7;

% Voxel size for the final map
mapVoxel = 0.01;    % m

% Geometric crop (ROI) of the final map, applied in section 11b.
% Coordinates in the map frame; use Inf/-Inf to leave an axis unbounded.
% The script prints the map extent before applying it, so the limits can be
% chosen on the first run with useMapROI = false.
% Examples: [-Inf Inf, -Inf Inf, -Inf 2.5] cuts only above 2.5 m
%           [0 40, -5 15, -Inf Inf]        isolates a segment in plan
useMapROI = true;
mapROI = [-Inf 40, ...     % X min max
          -1.8 2, ...     % Y min max
          -1 4];        % Z min max

% Odometry divergence detection (IMU/FAST-LIO losing tracking).
% The symptom is a position jump between two consecutive messages that is
% physically impossible for the sensor's speed: from that point onward the
% poses are no longer physical and must be discarded, not corrected.
odomMaxSpeed = 5.0;    % m/s, maximum plausible sensor speed
odomMaxJump  = 5.0;    % m, maximum absolute jump tolerated regardless of dt

% Saving the corrected map (Section 13), to disk as .pcd.
% The file name is generated at save time (Section 13) as a
% YY-MM-DD - HH-MM-SS timestamp, so different runs do not overwrite each other.
useSaveCorrectedMap = true;
savedBagDir = 'C:\Users\loren\Desktop\Measurment_v2\ClaudeCode\Thesis-final-wt2\3DModelPointCloudExtraction\SavedBag';

% Saving the corrected ODOMETRY (Section 13), as a rosbag2 carrying only
% /Odometry, plus a .csv of the same poses.
%
% The corrected .pcd is a merged map: it has no per-message poses, so nothing
% downstream that has to re-project a single scan into a camera frame can use
% it. WindowsDoorsDetection is exactly that case -- it reads /livox/lidar and
% /Odometry per frame. Writing the optimised trajectory back out as a bag is
% what lets that pipeline inherit these corrections, loop constraints
% included.
%
% The correction is carried to EVERY odometry message, not just to the
% keyframes: the per-keyframe correction transform C_k = A_opt_k / A_raw_k is
% interpolated (slerp on rotation, linear on translation) between the two
% keyframes bracketing each message, then applied to that message's raw pose.
% Writing keyframes alone would drop the trajectory from its full rate to one
% pose every keyframe, which is a real loss for de-skewing.
useSaveCorrectedOdom = true;
savedOdomDir = 'C:\Users\loren\Desktop\Measurment_v2\ClaudeCode\Thesis-final-wt2\3DModelPointCloudExtraction\SavedBag_Odometry';
odomOutTopic  = '/Odometry';
odomFrameId   = 'camera_init';   % FAST-LIO's map frame
odomChildFrameId = 'body';

%% 2. Bag reading
bag = ros2bagreader(bagPath);

selOdom  = select(bag, 'Topic', '/Odometry');
selCloud = select(bag, 'Topic', '/cloud_registered');

fprintf('Odometry:         %d messages\n', selOdom.NumMessages);
fprintf('cloud_registered: %d messages\n', selCloud.NumMessages);

odomMsgs  = readMessages(selOdom);
cloudMsgs = readMessages(selCloud);

%% 3. Conversion of odometry to homogeneous transforms
% nav_msgs/Odometry: pose.pose.position + pose.pose.orientation (quaternion)
nOdom = numel(odomMsgs);
posesRaw = repmat(rigidtform3d, nOdom, 1);
tOdom = zeros(nOdom, 1);

for i = 1:nOdom
    m = odomMsgs{i};
    p = m.pose.pose.position;
    q = m.pose.pose.orientation;

    % quat2rotm wants order [w x y z], ROS exposes x,y,z,w
    R = quat2rotm([q.w q.x q.y q.z]);
    t = [p.x p.y p.z];

    posesRaw(i) = rigidtform3d(R, t);
    tOdom(i) = double(m.header.stamp.sec) + double(m.header.stamp.nanosec)*1e-9;
end

%% 3b. Truncation at the first odometry divergence
% A position jump too large in too little time is not drift: it's the IMU
% losing tracking (reflective surface, bump, featureless stretch). From that
% message onward all poses are unreliable and are discarded, not "corrected"
% with loop closure.
trans = vertcat(posesRaw.Translation);
dStep = vecnorm(diff(trans), 2, 2);
dt    = diff(tOdom);
dt(dt <= 0) = eps;   % avoid division by zero/negative on non-monotonic timestamps
speedStep = dStep ./ dt;

divergeIdx = find(speedStep > odomMaxSpeed | dStep > odomMaxJump, 1, 'first');

if ~isempty(divergeIdx)
    fprintf(2, ['\nWARNING: odometry divergence detected at message %d/%d ' ...
        '(jump %.1f m in %.3f s, implied speed %.1f m/s).\n' ...
        'Poses truncated from here on: %d messages discarded out of %d.\n'], ...
        divergeIdx + 1, nOdom, dStep(divergeIdx), dt(divergeIdx), ...
        speedStep(divergeIdx), nOdom - divergeIdx, nOdom);

    posesRaw = posesRaw(1:divergeIdx);
    tOdom    = tOdom(1:divergeIdx);
    nOdom    = divergeIdx;
end

fprintf('Usable odometry duration: %.1f s\n', tOdom(end) - tOdom(1));

% Cloud timestamps, for association
nCloud = numel(cloudMsgs);
tCloud = zeros(nCloud, 1);
for i = 1:nCloud
    h = cloudMsgs{i}.header.stamp;
    tCloud(i) = double(h.sec) + double(h.nanosec)*1e-9;
end

%% 4. Cloud <-> pose association by timestamp
% Index alignment is not assumed: the pose closest in time to each cloud is
% searched, and pairings that are too far apart are discarded.
maxDt = 0.05;   % s
pairCloudIdx = [];
pairOdomIdx  = [];

for i = 1:nCloud
    [dt, j] = min(abs(tOdom - tCloud(i)));
    if dt <= maxDt
        pairCloudIdx(end+1) = i;   %#ok<SAGROW>
        pairOdomIdx(end+1)  = j;   %#ok<SAGROW>
    end
end

fprintf('Cloud/pose pairs associated: %d (discarded %d)\n', ...
    numel(pairCloudIdx), nCloud - numel(pairCloudIdx));

if isempty(pairCloudIdx)
    error(['No association found. Check that the timestamps of the two ' ...
        'topics are consistent, or raise maxDt.']);
end

%% 5. Keyframe selection
% A new keyframe when the translation or rotation threshold is exceeded.
kfSel = 1;   % the first is always a keyframe
lastT = posesRaw(pairOdomIdx(1)).Translation;
lastR = posesRaw(pairOdomIdx(1)).R;

for k = 2:numel(pairCloudIdx)
    T = posesRaw(pairOdomIdx(k)).Translation;
    R = posesRaw(pairOdomIdx(k)).R;

    dTrans = norm(T - lastT);
    % relative rotation angle, from the trace of the matrix
    dR = lastR' * R;
    dAng = abs(rad2deg(acos(max(-1, min(1, (trace(dR) - 1) / 2)))));

    if dTrans >= kfDistance || dAng >= kfAngle
        kfSel(end+1) = k;   %#ok<SAGROW>
        lastT = T;
        lastR = R;
    end
end

nKF = numel(kfSel);
fprintf('Keyframes selected: %d out of %d frames\n', nKF, numel(pairCloudIdx));

%% 5b. Cut the trajectory to a [start, end] window in seconds
% See note on useTimeCut/timeCutStartSec/timeCutEndSec in Section 1. Goes
% BEFORE cloud extraction (Section 6): the discarded keyframes are not even
% read from the bag, nor do they enter loop closure or the pose graph.
% Unlike a cut on a spatial coordinate, elapsed time is monotonic by
% construction: no risk of multiple or ambiguous crossings. NaN on one end
% leaves that side open.
if useTimeCut
    tKF = tOdom(pairOdomIdx(kfSel));
    elapsed = tKF - tKF(1);   % seconds from the start OF THE BAG, not of the window
    keep = true(size(elapsed));
    if ~isnan(timeCutStartSec), keep = keep & elapsed >= timeCutStartSec; end
    if ~isnan(timeCutEndSec),   keep = keep & elapsed <= timeCutEndSec;   end

    if all(keep)
        warning(['useTimeCut is active but the window [%.1f, %.1f] s covers the ' ...
            'entire trajectory (0-%.1f s): no cut applied.'], ...
            timeCutStartSec, timeCutEndSec, elapsed(end));
    elseif ~any(keep)
        error(['The window [%.1f, %.1f] s contains no keyframes (trajectory: ' ...
            '0-%.1f s). Check the limits.'], timeCutStartSec, timeCutEndSec, elapsed(end));
    else
        keepIdx = find(keep);
        fprintf('Trajectory cut: kept the keyframes between %.1f and %.1f s (%d/%d keyframes)\n', ...
            elapsed(keepIdx(1)), elapsed(keepIdx(end)), numel(keepIdx), numel(kfSel));
        kfSel = kfSel(keepIdx);
        nKF = numel(kfSel);
    end
end

%% 6. Cloud extraction in the body frame
% /cloud_registered is in the map frame: the pose is inverted to go back to
% the sensor frame, which is what both Scan Context (sensor-centric
% descriptor) and ICP need.
kfClouds = cell(nKF, 1);
kfPoses  = repmat(rigidtform3d, nKF, 1);

fprintf('Extracting keyframe clouds...\n');
nOutlierKFTotal = 0;
for k = 1:nKF
    ci = pairCloudIdx(kfSel(k));
    oi = pairOdomIdx(kfSel(k));

    xyz = rosReadXYZ(cloudMsgs{ci});
    xyz = xyz(all(isfinite(xyz), 2), :);
    pcMap = pointCloud(xyz);

    % un-transform: from the map frame to the body frame
    pcBody = pctransform(pcMap, invert(posesRaw(oi)));

    pcDown = pcdownsample(pcBody, 'gridAverage', kfVoxel);

    % Outlier filter (Section 1): after downsampling, so it's faster and the
    % uniform density of the voxel grid makes the neighbor statistics more
    % stable.
    if useOutlierFilterKF && pcDown.Count > outlierKFNumNeighbors
        nBefore = pcDown.Count;
        pcDown = pcdenoise(pcDown, ...
            'NumNeighbors', outlierKFNumNeighbors, ...
            'Threshold', outlierKFStdFactor);
        nOutlierKFTotal = nOutlierKFTotal + (nBefore - pcDown.Count);
    end

    kfClouds{k} = pcDown;
    kfPoses(k)  = posesRaw(oi);
end
if useOutlierFilterKF
    fprintf('  per-keyframe outlier filter: %d points removed in total\n', nOutlierKFTotal);
end

%% 6b. Realignment of attitude onto the floor plane
% FAST-LIO is gravity-aligned (the IMU observes it), so the floor normal,
% brought into the map frame, must remain vertical along the whole path.
% When it instead tilts progressively, that is attitude drift: the map
% "rotates" and a flat corridor appears to slope down.
%
% Loop closure does NOT see this drift: if the loops all fall within the
% same half of the path (before or after the drift event), each half stays
% consistent with itself and no constraint crosses the break. An external
% reference is needed, and the floor is one: it is a fixed physical
% direction, observed directly by the LiDAR.
%
% Roll and pitch are therefore imposed from the floor (2 DOF, drift-free by
% construction) and yaw and displacement are left to the odometry, which is
% reliable on those. The trajectory is re-integrated with the corrected
% attitudes.
if useGravityAlign
    fprintf('Realigning attitude on the floor...\n');

    % The search band starts from a low PERCENTILE, not from min(z): a
    % single spurious point below the floor would shift the band into empty
    % space and the fit would fail (with min(z) it failed on 25 out of 137
    % keyframes).
    nBody = nan(nKF, 3);
    % floorDist(k): signed distance origin-sensor -> floor plane, in the
    % BODY frame (kfClouds does not change with releveling: it is always
    % sensor-centric). Reused in Section 9 for the Z height constraint on
    % the trajectory, independent of the normal fit above: it is the known
    % term of the same plane, not a separate fit.
    floorDist = nan(nKF, 1);
    for k = 1:nKF
        loc = kfClouds{k}.Location;
        if size(loc,1) < 80, continue; end
        zref = prctile(loc(:,3), 2);
        cand = loc(loc(:,3) > zref - 0.12 & loc(:,3) < zref + floorBand, :);
        if size(cand,1) < 50, continue; end
        try
            [model, inl] = pcfitplane(pointCloud(cand), floorTol, [0 0 1], floorMaxTilt);
            if numel(inl) < 40, continue; end
        catch
            continue
        end
        n = model.Normal(:);
        if n(3) < 0, n = -n; end
        nBody(k,:) = n' / norm(n);
        floorDist(k) = -model.Parameters(4) / norm(model.Parameters(1:3));
    end
    validFloor = ~any(isnan(nBody), 2);
    fprintf('  valid floor normals: %d out of %d keyframes\n', nnz(validFloor), nKF);

    % Correction rotation where the floor was measured
    qC = nan(nKF, 4);
    tiltBefore = nan(nKF, 1);
    for k = 1:nKF
        if ~validFloor(k), continue; end
        nMap = kfPoses(k).R * nBody(k,:)';
        if nMap(3) < 0, nMap = -nMap; end
        nMap = nMap / norm(nMap);
        tiltBefore(k) = rad2deg(acos(max(-1, min(1, nMap(3)))));

        ax = cross(nMap, [0;0;1]);
        s  = norm(ax);
        c  = dot(nMap, [0;0;1]);
        if s > 1e-8
            ax  = ax / s;
            ang = atan2(s, c);
            K   = [0 -ax(3) ax(2); ax(3) 0 -ax(1); -ax(2) ax(1) 0];
            C   = eye(3) + sin(ang)*K + (1-cos(ang))*(K*K);   % Rodrigues
        else
            C = eye(3);
        end
        qC(k,:) = rotm2quat(C);
    end

    % In the gaps the correction is INTERPOLATED between the two valid
    % endpoints: freezing it at the last known value would let the error
    % accumulate again, and the gaps can be long (here up to 16 consecutive
    % keyframes).
    vi = find(validFloor);
    if isempty(vi)
        error(['No floor found in any keyframe: cannot realign. ' ...
            'Disable useGravityAlign or review floorBand.']);
    end
    for k = 1:nKF
        if validFloor(k), continue; end
        prev = vi(find(vi < k, 1, 'last'));
        next = vi(find(vi > k, 1, 'first'));
        if isempty(prev)
            qC(k,:) = qC(next,:);
        elseif isempty(next)
            qC(k,:) = qC(prev,:);
        else
            t = (k - prev) / (next - prev);
            qC(k,:) = slerpQuat(qC(prev,:), qC(next,:), t);
        end
    end

    Rc = cell(nKF,1);
    for k = 1:nKF
        Rc{k} = quat2rotm(qC(k,:)) * kfPoses(k).R;
    end

    % Re-integration of positions: the displacement in the body frame comes
    % from the odometry, the direction in which to apply it from the
    % corrected attitude.
    pOld = vertcat(kfPoses.Translation);
    pNew = zeros(nKF,3);
    pNew(1,:) = pOld(1,:);
    for k = 2:nKF
        dLocal = kfPoses(k-1).R' * (pOld(k,:) - pOld(k-1,:))';
        pNew(k,:) = pNew(k-1,:) + (Rc{k-1} * dLocal)';
    end

    for k = 1:nKF
        kfPoses(k) = rigidtform3d(Rc{k}, pNew(k,:));
    end

    % Residual tilt: this is the check that the correction did its job. If
    % it does not drop close to zero, the floor is not a good reference in
    % this environment (ramps, uneven terrain).
    tiltAfter = nan(nKF,1);
    for k = 1:nKF
        if ~validFloor(k), continue; end
        nMap = Rc{k} * nBody(k,:)';
        if nMap(3) < 0, nMap = -nMap; end
        tiltAfter(k) = rad2deg(acos(max(-1, min(1, nMap(3)))));
    end

    fprintf('  floor tilt: median %.2f -> %.2f deg, max %.2f -> %.2f deg\n', ...
        median(tiltBefore(~isnan(tiltBefore))), median(tiltAfter(~isnan(tiltAfter))), ...
        max(tiltBefore), max(tiltAfter));
    fprintf('  trajectory Z drift: %.2f m -> %.2f m\n', ...
        max(pOld(:,3))-min(pOld(:,3)), max(pNew(:,3))-min(pNew(:,3)));
end

%% 6c. YAW drift correction on the wall direction
% The floor constrains only 2 of 3 DOF: the normal of a horizontal plane is
% invariant to rotation about the vertical axis, so it tells "up" but not
% "north". Yaw remains free to drift, and the symptom is that corridors no
% longer meet at right angles in plan.
%
% The reference for yaw is the WALLS: the near-horizontal normals are taken
% and their dominant azimuth is computed, folded modulo 90 degrees (so the
% four sides of an orthogonal corridor fall on the same value). The
% correction brings that azimuth back to the reference value.
%
% WARNING, this is NOT a pure measurement like the floor: it assumes the
% building has a coherent dominant direction (parallel or perpendicular
% walls). If the building really had a wing at 45 degrees, this correction
% would wrongly straighten it, falsifying the geometry. Always check the two
% control prints: if the dispersion does NOT drop clearly, the walls do not
% belong to a single orthogonal family and the assumption does not hold for
% this building: in that case, disable it.
if useYawAlign
    fprintf('Correcting yaw drift on the walls...\n');

    azi = nan(nKF,1);
    for k = 1:nKF
        pc = kfClouds{k};
        if pc.Count < 200, continue; end
        try
            nrm = pcnormals(pc, 20);
        catch
            continue
        end
        nMap   = (kfPoses(k).R * nrm')';
        isWall = abs(nMap(:,3)) < wallMaxNz;     % horizontal normal => wall
        if nnz(isWall) < 50, continue; end
        a = atan2(nMap(isWall,2), nMap(isWall,1));
        % x4 brings the period from 90 to 360 degrees: this way the circular
        % mean is well defined and does not suffer from the 0/90 wrap.
        azi(k) = mod(rad2deg(angle(mean(exp(1i*4*a))))/4, 90);
    end
    validWall = ~isnan(azi);
    fprintf('  azimuth estimated on %d out of %d keyframes\n', nnz(validWall), nKF);

    if nnz(validWall) < 10
        warning(['Too few keyframes with walls: yaw correction skipped. ' ...
            'Environment is probably open or poorly structured.']);
    else
        z = nan(nKF,1) + 1i*nan;
        z(validWall) = exp(1i*4*deg2rad(azi(validWall)));

        % Reference: circular mean of the first keyframes, before drift
        % manifests. Anchoring to the start preserves the original
        % orientation of the map.
        nRef = min(yawRefKF, nKF);
        zRef = mean(z(1:nRef), 'omitnan');

        % Circular smoothing: the per-keyframe estimate is noisy and should
        % not be chased, drift is a slow phenomenon.
        zS = nan(nKF,1) + 1i*nan;
        hw = floor(yawSmooth/2);
        for k = 1:nKF
            w = z(max(1,k-hw):min(nKF,k+hw));
            w = w(~isnan(w));
            if ~isempty(w), zS(k) = mean(w); end
        end
        vw = find(~isnan(zS));
        for k = 1:nKF
            if ~isnan(zS(k)), continue; end
            [~, i] = min(abs(vw - k));
            zS(k) = zS(vw(i));
        end

        dYaw = zeros(nKF,1);
        for k = 1:nKF
            d = rad2deg(angle(zS(k) / zRef)) / 4;
            dYaw(k) = mod(d + 45, 90) - 45;     % wrap into [-45, 45]
        end

        pOldY = vertcat(kfPoses.Translation);
        RcY   = cell(nKF,1);
        for k = 1:nKF
            th = -deg2rad(dYaw(k));
            Cz = [cos(th) -sin(th) 0; sin(th) cos(th) 0; 0 0 1];
            RcY{k} = Cz * kfPoses(k).R;
        end

        pNewY = zeros(nKF,3);
        pNewY(1,:) = pOldY(1,:);
        for k = 2:nKF
            dLocal = kfPoses(k-1).R' * (pOldY(k,:) - pOldY(k-1,:))';
            pNewY(k,:) = pNewY(k-1,:) + (RcY{k-1}*dLocal)';
        end

        for k = 1:nKF
            kfPoses(k) = rigidtform3d(RcY{k}, pNewY(k,:));
        end

        % Check: the circular dispersion must drop clearly.
        aziAfter = nan(nKF,1);
        for k = 1:nKF
            pc = kfClouds{k};
            if pc.Count < 200, continue; end
            try
                nrm = pcnormals(pc, 20);
            catch
                continue
            end
            nMap   = (RcY{k} * nrm')';
            isWall = abs(nMap(:,3)) < wallMaxNz;
            if nnz(isWall) < 50, continue; end
            a = atan2(nMap(isWall,2), nMap(isWall,1));
            aziAfter(k) = mod(rad2deg(angle(mean(exp(1i*4*a))))/4, 90);
        end
        cdisp = @(v) 1 - abs(mean(exp(1i*4*deg2rad(v(~isnan(v))))));

        fprintf('  correction applied: from %.2f to %.2f deg\n', min(dYaw), max(dYaw));
        fprintf('  wall direction dispersion: %.3f -> %.3f  (lower = more coherent)\n', ...
            cdisp(azi), cdisp(aziAfter));
    end
end

%% 6d. Topological classification of keyframes (node/line)
% After 6b/6c, kfPoses is in its final form: it is the geometry that loop
% closure will actually use, so the classification must be done here, not
% before. See note on nodeAngWindow/nodeAngThreshold in Section 1.
[isNodeKF, cumAngKF] = classifyKfNodes(kfPoses, nodeAngWindow, nodeAngThreshold);

fprintf('\n--- Keyframe topological classification ---\n');
fprintf('Node (turn):    %d out of %d (%.0f%%)\n', nnz(isNodeKF), nKF, 100*nnz(isNodeKF)/nKF);
fprintf('Line (straight): %d out of %d (%.0f%%)\n', nnz(~isNodeKF), nKF, 100*nnz(~isNodeKF)/nKF);

%% 6e. Direct correction of Z height from the floor (BEFORE Scan Context/ICP)
% floorDist (Section 6b) is a local measurement, unaffected by accumulated Z
% drift: here it is applied directly to kfPoses, not only as an edge in the
% pose graph (Section 9). Reason: Arel = kfPoses(i).A \ kfPoses(j).A in
% Section 8 is the INITIAL guess for ICP on the loop candidates. With the
% raw Z drift (up to ~1.8 m on this dataset) inside that initial guess, ICP
% can end up outside the convergence basin on candidates that are otherwise
% valid (see the Section 8 comment on "Tinit too far off"). Correcting here
% gives ICP a better starting estimate; the constraint in the pose graph
% (Section 9) is still useful to defend this height during loop optimization.
%
% Note: after leveling roll/pitch (6b), the rotations in 6c are pure
% rotation about Z (yaw): they do not mix the Z component of translation, so
% applying this correction here (after 6c) instead of before gives the same
% numerical result, with less risk of interfering with the re-integration
% in 6c.
floorFitRate = nnz(validFloor) / nKF;
useGravityZ  = useGravityAlign && floorFitRate >= floorFitMin && validFloor(1);
if useGravityZ
    pZ = vertcat(kfPoses.Translation);
    floorWorldZPre = pZ(:,3) - floorDist;
    fprintf('\nFloor height (pre-correction): median %.3f m, std %.3f m (on %d/%d keyframes)\n', ...
        median(floorWorldZPre(validFloor)), std(floorWorldZPre(validFloor)), nnz(validFloor), nKF);

    z1 = kfPoses(1).Translation(3);
    for k = 1:nKF
        if ~validFloor(k), continue; end
        p = kfPoses(k).Translation;
        p(3) = z1 + (floorDist(k) - floorDist(1));
        kfPoses(k) = rigidtform3d(kfPoses(k).R, p);
    end

    pZ = vertcat(kfPoses.Translation);
    floorWorldZPost = pZ(:,3) - floorDist;
    fprintf('Floor height (post-correction): median %.3f m, std %.3f m (on %d/%d keyframes)\n', ...
        median(floorWorldZPost(validFloor)), std(floorWorldZPost(validFloor)), nnz(validFloor), nKF);
    fprintf('Direct Z correction applied to %d/%d keyframes (%.0f%% floor detection weight)\n', ...
        nnz(validFloor), nKF, 100*floorFitRate);
else
    fprintf(['\nFloor detected in %.0f%% of keyframes (< %.0f%% required) or node 1 has no fit: ' ...
        'no direct Z correction\n'], 100*floorFitRate, 100*floorFitMin);
end

%% 7. Loop detection with Scan Context
% NOTE: check the signatures with "doc scanContextLoopDetector" if MATLAB
% reports invalid arguments. Parameter names have changed between releases.
fprintf('Computing Scan Context descriptors...\n');

loopDetector = scanContextLoopDetector;
loopCandidates = [];   % [from to]

for k = 1:nKF
    descriptor = scanContextDescriptor(kfClouds{k});

    if k > scNumExcluded
        [loopIds, ~] = detectLoop(loopDetector, descriptor, ...
            'DistanceThreshold', scDistThreshold, ...
            'NumExcludedDescriptors', scNumExcluded, ...
            'MaxDetections', scMaxDetections);

        for m = 1:numel(loopIds)
            loopCandidates(end+1, :) = [loopIds(m), k];   %#ok<SAGROW>
        end
    end

    addDescriptor(loopDetector, k, descriptor);
end

fprintf('Loop candidates from Scan Context: %d\n', size(loopCandidates, 1));

%% 7b. Revisit diagnostics and candidates by spatial proximity
% Scan Context compares the appearance of the scan: in a corridor, where
% every stretch resembles every other, it is not very reliable. Complementary
% criterion: two keyframes far apart in time but close in space are, most
% likely, a revisit.
%
% Distance is measured in the XY plane ONLY. The drift here is mainly in Z
% (the corridor "rises"): including Z would artificially push apart the very
% revisits we are looking for.

kfXYZ = vertcat(kfPoses.Translation);

% XY distance matrix without pdist2 (avoids the Statistics Toolbox)
dx  = kfXYZ(:,1) - kfXYZ(:,1)';
dy  = kfXYZ(:,2) - kfXYZ(:,2)';
dXY = sqrt(dx.^2 + dy.^2);

[II, JJ] = ndgrid(1:nKF, 1:nKF);
gapIdx = abs(II - JJ);

revisit = triu(dXY <= proxRadius & gapIdx >= proxMinGap, 1);

spanXYZ = max(kfXYZ, [], 1) - min(kfXYZ, [], 1);

fprintf('\n--- Revisit diagnostics ---\n');
fprintf('Path length:         %.1f m\n', sum(vecnorm(diff(kfXYZ), 2, 2)));
fprintf('XY extent:           %.1f x %.1f m\n', spanXYZ(1), spanXYZ(2));
fprintf('Pose Z extent:       %.2f m  (suspected drift)\n', spanXYZ(3));
fprintf('Pairs at >=%d keyframes apart and <=%.1f m in XY: %d\n', ...
    proxMinGap, proxRadius, nnz(revisit));

[ri, rj] = find(revisit);
proxCandidates = [ri, rj];

% If there are very many revisits (path overlapping for a long stretch),
% only the tightest are kept, otherwise ICP becomes prohibitive.
if size(proxCandidates, 1) > proxMaxCand
    dSel = dXY(sub2ind([nKF nKF], ri, rj));
    [~, ord] = sort(dSel, 'ascend');
    proxCandidates = proxCandidates(ord(1:proxMaxCand), :);
    fprintf('  (reduced to the %d closest)\n', proxMaxCand);
end

if isempty(proxCandidates)
    fprintf(2, ['\nWARNING: the path NEVER revisits itself.\n' ...
        'Loop closure cannot work, on any axis: there is no\n' ...
        'observation linking two distant points of the path,\n' ...
        'so the graph has no way to "know" that the height is wrong.\n' ...
        'A graph with only sequential constraints has as its exact optimum\n' ...
        'the starting odometry: that is why BEFORE and AFTER coincide.\n' ...
        'Straightening this map requires an external constraint (e.g. the\n' ...
        'floor plane observed by the LiDAR), not a loop closure.\n']);
end

% Union of the two candidate sets
if isempty(loopCandidates)
    allCandidates = proxCandidates;
elseif isempty(proxCandidates)
    allCandidates = loopCandidates;
else
    allCandidates = unique([double(loopCandidates); proxCandidates], 'rows');
end
fprintf('Total candidates to verify: %d\n\n', size(allCandidates, 1));

% Node/line classification per candidate (see Section 6d): this does not
% decide anything yet, it is only used to read the Section 8 ICP results
% split by keyframe type.
topoLabel = ["line" "node"];   % topoLabel(isNodeKF(k)+1) -> "line"/"node"
fprintf('--- Candidates with topological classification ---\n');
fprintf('%5s %5s  %-6s %-6s  %6s %6s\n', 'i', 'j', 'topo_i', 'topo_j', 'cumA_i', 'cumA_j');
for c = 1:size(allCandidates, 1)
    i = allCandidates(c, 1);
    j = allCandidates(c, 2);
    fprintf('%5d %5d  %-6s %-6s  %6.1f %6.1f\n', i, j, ...
        topoLabel(isNodeKF(i)+1), topoLabel(isNodeKF(j)+1), cumAngKF(i), cumAngKF(j));
end
fprintf('\n');

%% 8. Loop verification with ICP
% Scan Context produces false positives, typically in repetitive
% environments like corridors. ICP discards them: if the two scans do not
% actually align, the RMSE stays high.
loopConstraints = {};   % {fromIdx, toIdx, relative rigidtform3d, rmse}

fprintf('ICP verification of candidates...\n');
for c = 1:size(allCandidates, 1)
    i = allCandidates(c, 1);
    j = allCandidates(c, 2);

    % Initial estimate from odometry: relative pose from i to j.
    Arel = kfPoses(i).A \ kfPoses(j).A;

    % On a real revisit this estimate is polluted precisely by the drift we
    % want to correct: if the height is off by meters, ICP starts outside
    % the convergence basin and always fails. A variant with the Z
    % component zeroed is therefore also tried, and the better of the two
    % is kept.
    ArelFlat = Arel;
    ArelFlat(3, 4) = 0;

    initGuesses = {Arel, ArelFlat};
    bestRmse  = inf;
    bestTform = [];

    % Coarse-to-fine ICP: with an initial estimate wrong by meters (the
    % drift we want to correct), a tight 'InlierDistance' finds too few
    % correspondences and never converges: it fails even when the two
    % clouds actually do overlap. A first pass with wide correspondences
    % brings the alignment into the right convergence basin; the second,
    % tight pass, refines it.
    coarseInlierDistance = max(icpMaxDistance * 6, 6.0);

    for g = 1:numel(initGuesses)
        try
            [tfCoarse, ~, ~] = pcregistericp(kfClouds{j}, kfClouds{i}, ...
                'InitialTransform', rigidtform3d(initGuesses{g}), ...
                'InlierDistance', coarseInlierDistance, ...
                'MaxIterations', 50);

            [tf, ~, rmse] = pcregistericp(kfClouds{j}, kfClouds{i}, ...
                'InitialTransform', tfCoarse, ...
                'InlierDistance', icpMaxDistance);

            if rmse < bestRmse
                bestRmse  = rmse;
                bestTform = tf;
            end
        catch ME
            fprintf('  ICP failed on %d -> %d: %s\n', i, j, ME.message);
        end
    end

    if isempty(bestTform)
        continue
    end

    % 6-DOF discrepancy between the ICP measurement and the odometry estimate
    Terr     = Arel \ bestTform.A;
    transErr = norm(Terr(1:3, 4));
    rotErr   = abs(rad2deg(acos(max(-1, min(1, (trace(Terr(1:3,1:3)) - 1) / 2)))));

    % Node/line label of the candidate (Section 6d), diagnostic only: it
    % does not yet influence acceptance or weights.
    topoTag = sprintf('[%s-%s]', topoLabel(isNodeKF(i)+1), topoLabel(isNodeKF(j)+1));

    if bestRmse > icpMaxRMSE
        fprintf('  loop DISCARDED  %d -> %d  (rmse %.3f m, above threshold)  %s\n', ...
            i, j, bestRmse, topoTag);
    elseif transErr > loopMaxTransErr || rotErr > loopMaxRotErr
        fprintf(['  loop DISCARDED  %d -> %d  (rmse %.3f m ok, but contradicts ' ...
            'the odometry by %.1f m / %.1f deg: false positive)  %s\n'], ...
            i, j, bestRmse, transErr, rotErr, topoTag);
    else
        loopConstraints{end+1} = {i, j, bestTform, bestRmse};   %#ok<SAGROW>
        fprintf('  loop accepted %d -> %d  (rmse %.3f m, discrepancy %.2f m / %.1f deg)  %s\n', ...
            i, j, bestRmse, transErr, rotErr, topoTag);
    end
end

nLoops = numel(loopConstraints);
fprintf('Loops verified and accepted: %d\n', nLoops);

if nLoops == 0
    if isempty(proxCandidates)
        warning(['No loop accepted AND no geometric revisit: ' ...
            'the path never retraces its steps. The map CANNOT be ' ...
            'corrected with loop closure. See the diagnostics above.']);
    else
        warning(['No loop accepted, but %d geometric revisits ' ...
            'exist: it is ICP rejecting them. The thresholds are too ' ...
            'tight, or the initial estimate is already too far off ' ...
            '(with %.1f m of Z drift, Tinit from odometry can be ' ...
            'outside the convergence basin). Try raising ' ...
            'icpMaxRMSE and icpMaxDistance.'], ...
            size(proxCandidates, 1), spanXYZ(3));
    end
end

% Checkpoint: the above (bag reading, keyframes, ICP) is expensive and does
% not depend on the pose graph weights. Saved here so the weights can be
% iterated on without redoing everything from scratch.
checkpointFile = fullfile(fileparts(bagPath), 'loop_closure_checkpoint.mat');
save(checkpointFile, 'kfPoses', 'loopConstraints', 'nLoops', 'nKF', ...
    'kfVoxel', 'mapVoxel', 'kfClouds', 'pairCloudIdx', 'pairOdomIdx', 'kfSel', '-v7.3');
fprintf('Checkpoint saved to: %s\n', checkpointFile);

%% 9. Building the pose graph
pg = poseGraph3D;

% Information matrix: 21 elements, upper triangle of a 6x6.
% Diagonal = [x y z rx ry rz], higher values = stiffer constraint.
infoVecOdom = buildInfoVector(infoOdom);

% Sequential constraints from odometry
for k = 2:nKF
    Trel = kfPoses(k-1).A \ kfPoses(k).A;
    addRelativePose(pg, tform2measurement(Trel), infoVecOdom, k-1, k);
end

% Loop constraints: individual weight scaled on the actual ICP RMSE, not
% fixed. A loop with rmse << sigma0 is a very reliable measurement and
% should weigh more than the sheer multitude of odometry constraints; a
% loop close to the acceptance threshold weighs less.
if ~useLoopClosure
    fprintf(['Loop closure DISABLED (useLoopClosure = false): the graph uses\n' ...
        'only the sequential constraints on the poses already realigned to gravity.\n']);
    nLoops = 0;
end

fprintf('Loop constraint weights (infoOdom = %d for comparison):\n', infoOdom);
for c = 1:nLoops
    i     = loopConstraints{c}{1};
    j     = loopConstraints{c}{2};
    tform = loopConstraints{c}{3};
    rmse  = loopConstraints{c}{4};

    % Topological weight (Section 6d): node-node more reliable, line-line
    % less, node-line unchanged relative to the previous behavior.
    if isNodeKF(i) && isNodeKF(j)
        topoW = topoWNodeNode;
    elseif isNodeKF(i) || isNodeKF(j)
        topoW = topoWMixed;
    else
        topoW = topoWLineLine;
    end

    infoLoopEff = topoW * infoLoop * (sigma0Loop / max(rmse, sigma0Loop/2))^2;
    fprintf('  %d -> %d : rmse %.3f m, topo x%.1f -> info %.0f\n', i, j, rmse, topoW, infoLoopEff);

    infoVecLoopC = buildInfoVector(infoLoopEff);
    addRelativePose(pg, tform2measurement(tform.A), infoVecLoopC, i, j);
end

% Gravity constraints.
% Without these, the realignment from section 6b is partly UNDONE by the
% optimization: the graph contains nothing that says "the floor is
% horizontal", so the optimizer is free to re-tilt the attitude in order to
% satisfy the loops.
%
% poseGraph3D has no unary factors (priors), but node 1 is fixed by the
% optimizer: a 1->k constraint with a measurement equal to the desired
% absolute pose of k behaves like a prior on k. To constrain ONLY roll and
% pitch, leaving x, y, z and yaw free, an anisotropic information matrix is
% used: high weight on rx,ry and near-zero (but positive, must remain
% positive definite) weight on the other DOFs.
if useGravityAlign && useGravityFactor
    % useGravityZ/floorFitRate/validFloor already computed in Section 6e,
    % the same correction already applied directly to kfPoses there. Here it
    % is only repeated as an edge in the pose graph, to defend that height
    % during loop optimization (the optimizer could otherwise partly remove
    % it to satisfy the other constraints).
    if useGravityZ
        fprintf('Z constraint in the pose graph enabled (weight %g)\n', infoGravZ);
    else
        fprintf('No Z constraint in the pose graph (see Section 6e)\n');
    end

    T0inv = kfPoses(1).A \ eye(4);
    nZCorr = 0;
    for k = 2:nKF
        Ak = T0inv * kfPoses(k).A;    % pose of k relative to node 1, already leveled
        wZ = infoGravFree;
        if useGravityZ && validFloor(k)
            Ak(3,4) = floorDist(k) - floorDist(1);   % target Z from the floor, not from poseZ (no drift)
            wZ = infoGravZ;
            nZCorr = nZCorr + 1;
        end
        addRelativePose(pg, tform2measurement(Ak), ...
            buildInfoVectorAniso(infoGravFree, infoGravRP, wZ), 1, k);
    end
    fprintf('Gravity constraints added: %d (roll/pitch weight %g, Z from floor on %d/%d)\n', ...
        nKF-1, infoGravRP, nZCorr, nKF-1);
end

fprintf('\nPose graph: %d nodes, %d constraints (%d loops)\n', ...
    pg.NumNodes, pg.NumEdges, nLoops);

%% 10. Optimization
fprintf('Optimizing...\n');
pgOpt = optimizePoseGraph(pg, 'builtin-trust-region');

%% 11. Map reconstruction with corrected poses
% IMPORTANT: poseGraph3D ALWAYS anchors node 1 at the origin with identity
% orientation, while kfPoses(1) has its own position and attitude. Without
% bringing the result back to the starting frame, "before" and "after" live
% in two different global frames, rotated relative to each other: comparing
% the Z spans becomes meaningless (a global rotation changes the span even
% with an identical trajectory) and the map comes out rotated relative to
% the original.
nodesOpt = nodeEstimates(pgOpt);

T0 = kfPoses(1).A;               % frame of the first keyframe
posesOpt = repmat(rigidtform3d, nKF, 1);
for k = 1:nKF
    n  = nodesOpt(k, :);         % nodeEstimates returns [x y z qw qx qy qz]
    Ak = eye(4);
    Ak(1:3,1:3) = quat2rotm(n(4:7));
    Ak(1:3,4)   = n(1:3)';
    Ak = T0 * Ak;                % bring back to the starting frame
    posesOpt(k) = rigidtform3d(Ak(1:3,1:3), Ak(1:3,4)');
end
nodesOpt = [vertcat(posesOpt.Translation), ...
            cell2mat(arrayfun(@(p) rotm2quat(p.R), posesOpt, 'UniformOutput', false))];

allXYZ = cell(nKF, 1);
for k = 1:nKF
    pcT = pctransform(kfClouds{k}, posesOpt(k));
    allXYZ{k} = pcT.Location;
end

pcOpt = pointCloud(vertcat(allXYZ{:}));
pcOpt = pcdownsample(pcOpt, 'gridAverage', mapVoxel);

% Original map, for comparison
allXYZraw = cell(nKF, 1);
for k = 1:nKF
    pcT = pctransform(kfClouds{k}, kfPoses(k));
    allXYZraw{k} = pcT.Location;
end
pcRaw = pointCloud(vertcat(allXYZraw{:}));
pcRaw = pcdownsample(pcRaw, 'gridAverage', mapVoxel);

% Outlier filter on the final map (Section 1): here "seam" outliers between
% different scans also appear, which do not exist in the single keyframe.
% Applied to both maps, for a consistent comparison.
if useOutlierFilterMap
    nOptBefore = pcOpt.Count;
    nRawBefore = pcRaw.Count;
    pcOpt = pcdenoise(pcOpt, 'NumNeighbors', outlierMapNumNeighbors, 'Threshold', outlierMapStdFactor);
    pcRaw = pcdenoise(pcRaw, 'NumNeighbors', outlierMapNumNeighbors, 'Threshold', outlierMapStdFactor);
    fprintf('\nOutlier filter on the map:\n');
    fprintf('  corrected map: %d -> %d points (%.1f%% removed)\n', ...
        nOptBefore, pcOpt.Count, 100*(nOptBefore - pcOpt.Count)/nOptBefore);
    fprintf('  raw map:       %d -> %d points (%.1f%% removed)\n', ...
        nRawBefore, pcRaw.Count, 100*(nRawBefore - pcRaw.Count)/nRawBefore);
end

%% 11b. Geometric crop (ROI) on the final map
% WATCH the point at which this is applied. The ROI goes HERE, on the
% already reconstructed map, not on the keyframe clouds: those are in the
% body frame and are used by Scan Context (descriptor that wants the whole
% scan), by ICP (which needs geometry to lock onto), and by the floor fit
% (section 6b). Cropping them would degrade registration and realignment.
%
% The coordinates are in the map frame, the same as the trajectory: use the
% extent printed below to choose the limits. Inf/-Inf leave the axis
% unbounded.
fprintf('\n--- Map extent (to choose the ROI) ---\n');
fprintf('  X: %7.2f  %7.2f\n', pcOpt.XLimits);
fprintf('  Y: %7.2f  %7.2f\n', pcOpt.YLimits);
fprintf('  Z: %7.2f  %7.2f\n', pcOpt.ZLimits);

if useMapROI
    nBeforeOpt = pcOpt.Count;
    nBeforeRaw = pcRaw.Count;

    % The crop is applied to BOTH maps: comparing a cropped area with the
    % whole map would make the before/after comparison meaningless.
    pcOpt = select(pcOpt, findPointsInROI(pcOpt, mapROI));
    pcRaw = select(pcRaw, findPointsInROI(pcRaw, mapROI));

    fprintf('ROI applied [%g %g, %g %g, %g %g]\n', mapROI);
    fprintf('  corrected map: %d -> %d points (%.1f%% removed)\n', ...
        nBeforeOpt, pcOpt.Count, 100*(nBeforeOpt - pcOpt.Count)/nBeforeOpt);
    fprintf('  raw map:       %d -> %d points (%.1f%% removed)\n', ...
        nBeforeRaw, pcRaw.Count, 100*(nBeforeRaw - pcRaw.Count)/nBeforeRaw);

    if pcOpt.Count == 0 || pcRaw.Count == 0
        error(['The ROI emptied the map: no points inside the limits. ' ...
            'Check that the coordinates are in the map frame printed above.']);
    end
end

%% 12. Comparison
% WATCH the metric. The Z extent of the MAP does not measure drift: each
% single scan already covers several meters vertically (the LiDAR sees
% floor and ceiling, and much higher in atriums), so the bounding box stays
% large even with a perfect trajectory. Drift lives in the POSES, and that
% is where it must be measured.
zTrajRaw = vertcat(kfPoses.Translation);
zTrajRaw = zTrajRaw(:,3);
zTrajOpt = nodesOpt(:,3);

spanTrajRaw = max(zTrajRaw) - min(zTrajRaw);
spanTrajOpt = max(zTrajOpt) - min(zTrajOpt);

fprintf('\n--- Vertical drift of the TRAJECTORY (correct metric) ---\n');
fprintf('Before: pose Z extent %.2f m\n', spanTrajRaw);
fprintf('After:  pose Z extent %.2f m  (%+.1f%%)\n', ...
    spanTrajOpt, 100*(spanTrajOpt - spanTrajRaw)/spanTrajRaw);
fprintf('Mean node displacement: %.3f m\n', ...
    mean(vecnorm(nodesOpt(:,1:3) - vertcat(kfPoses.Translation), 2, 2)));

% Z extent of the map, reported only as a reference: NOT a drift indicator,
% see comment above.
fprintf('\n--- Map Z extent (NOT a drift indicator) ---\n');
fprintf('Before: %7.2f  %7.2f   (extent %.2f m)\n', ...
    pcRaw.ZLimits, diff(pcRaw.ZLimits));
fprintf('After:  %7.2f  %7.2f   (extent %.2f m)\n', ...
    pcOpt.ZLimits, diff(pcOpt.ZLimits));

figure('Color', 'k', 'Name', 'Before/after loop closure comparison - BEFORE');
pcshow(pcRaw, 'MarkerSize', 20);
title(sprintf('BEFORE, Z extent %.2f m', diff(pcRaw.ZLimits)), 'Color', 'w');
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
axis equal; colormap(gca, turbo);

figure('Color', 'k', 'Name', 'Before/after loop closure comparison - AFTER');
pcshow(pcOpt, 'MarkerSize', 20);
title(sprintf('AFTER, Z extent %.2f m', diff(pcOpt.ZLimits)), 'Color', 'w');
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
axis equal; colormap(gca, turbo);

% Trajectory before and after
figure('Name', 'Trajectory');
trajRaw = vertcat(kfPoses.Translation);
plot3(trajRaw(:,1), trajRaw(:,2), trajRaw(:,3), 'r-', 'LineWidth', 1.5);
hold on;
plot3(nodesOpt(:,1), nodesOpt(:,2), nodesOpt(:,3), 'g-', 'LineWidth', 1.5);
legend('Before', 'After', 'Location', 'best');
xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
title('Keyframe trajectory');
axis equal; grid on;

%% 13. Saving
% See useSaveCorrectedMap / savedBagDir in Section 1. File name = save
% timestamp (YY-MM-DD - HH-MM-SS), so different runs do not overwrite each
% other.
if useSaveCorrectedMap
    if ~exist(savedBagDir, 'dir')
        mkdir(savedBagDir);
    end
    stamp = char(datetime('now', 'Format', 'yy-MM-dd - HH-mm-ss'));
    correctedMapOutFile = fullfile(savedBagDir, [stamp '.pcd']);
    pcwrite(pcOpt, correctedMapOutFile, 'Encoding', 'binary');
    fprintf('\nCorrected map saved to:\n  %s\n', correctedMapOutFile);
else
    fprintf('\nMap saving disabled (useSaveCorrectedMap = false)\n');
end

% Corrected odometry: see useSaveCorrectedOdom / savedOdomDir in Section 1.
% Shares the same timestamp as the .pcd above when both are written in one
% run, so a map and the trajectory that produced it stay identifiable as a
% pair.
if useSaveCorrectedOdom
    if ~exist('stamp', 'var')
        stamp = char(datetime('now', 'Format', 'yy-MM-dd - HH-mm-ss'));
    end
    writeCorrectedOdometry(savedOdomDir, stamp, ...
        tOdom, posesRaw, ...
        tOdom(pairOdomIdx(kfSel)), posesRaw(pairOdomIdx(kfSel)), posesOpt, ...
        odomOutTopic, odomFrameId, odomChildFrameId);
else
    fprintf('Odometry saving disabled (useSaveCorrectedOdom = false)\n');
end

%% Support functions
function [isNode, cumAng] = classifyKfNodes(kfPoses, winHalf, angThresh)
    % Node/line proxy (Xu et al. 2024, adapted): dAng between consecutive
    % keyframes (same formula as Section 5, but between kfPoses instead of
    % raw frames), summed in absolute value over a window of radius winHalf
    % to capture turns spread over several steps. Absolute sum, not
    % mean/net rotation: an S-curve must not cancel out.
    n = numel(kfPoses);
    dAng = zeros(n, 1);   % dAng(k) = rotation from kfPoses(k-1) to kfPoses(k)
    for k = 2:n
        dR = kfPoses(k-1).R' * kfPoses(k).R;
        dAng(k) = abs(rad2deg(acos(max(-1, min(1, (trace(dR) - 1) / 2)))));
    end

    cumAng = zeros(n, 1);
    for k = 1:n
        w = dAng(max(1,k-winHalf):min(n,k+winHalf));
        cumAng(k) = sum(w);
    end

    isNode = cumAng >= angThresh;
end

function v = buildInfoVectorAniso(wFree, wRP, wZ)
    % Anisotropic information matrix: weight on roll and pitch, near-zero on
    % the rest. The diagonal order is [x y z rx ry rz]. wFree must be > 0:
    % addRelativePose rejects non positive-definite matrices. wZ optional
    % (default wFree): weight on Z when the height constraint (Section 9) is
    % active for the current keyframe.
    if nargin < 3, wZ = wFree; end
    M = diag([wFree wFree wZ wRP wRP wFree]);
    v = zeros(1, 21);
    n = 0;
    for i = 1:6
        for j = i:6
            n = n + 1;
            v(n) = M(i, j);
        end
    end
end

function v = buildInfoVector(w)
    % Diagonal 6x6 information matrix, returned as the 21 elements of the
    % upper triangle in the order required by addRelativePose.
    % The order is row-major: (1,1)...(1,6),(2,2)...(2,6),...,(6,6).
    % MATLAB's triu with linear indexing is column-major, so it would put
    % the diagonal in the wrong positions.
    M = diag([w w w w w w]);
    v = zeros(1, 21);
    n = 0;
    for i = 1:6
        for j = i:6
            n = n + 1;
            v(n) = M(i, j);
        end
    end
end

function meas = tform2measurement(A)
    % From a 4x4 homogeneous matrix to [x y z qw qx qy qz]
    R = A(1:3, 1:3);
    t = A(1:3, 4)';
    q = rotm2quat(R);
    meas = [t q];
end

function q = slerpQuat(q0, q1, t)
    % Spherical interpolation between two quaternions. Written by hand
    % because quatinterp requires the Aerospace Toolbox, which is not among
    % the requirements.
    q0 = q0 / norm(q0);
    q1 = q1 / norm(q1);

    c = dot(q0, q1);
    if c < 0            % shortest path on the sphere
        q1 = -q1;
        c  = -c;
    end

    if c > 0.9995       % nearly aligned: linear, avoids the unstable division
        q = q0 + t*(q1 - q0);
        q = q / norm(q);
        return
    end

    th0 = acos(max(-1, min(1, c)));
    th  = th0 * t;
    q2  = q1 - q0*c;
    q2  = q2 / norm(q2);
    q   = q0*cos(th) + q2*sin(th);
end

function writeCorrectedOdometry(savedOdomDir, stamp, tOdom, posesRaw, ...
                                tKF, posesKFRaw, posesOpt, ...
                                topicName, frameId, childFrameId)
    % Corrected /Odometry to a rosbag2 (plus a .csv of the same poses).
    %
    % The pose graph optimises KEYFRAMES only, but the consumers of this bag
    % interpolate a pose per LiDAR point, so a keyframe-rate trajectory would
    % throw away most of the de-skewing. The correction is therefore carried to
    % every odometry message: for keyframe k the correction in the map frame is
    %
    %     C_k = A_opt_k * inv(A_raw_k)
    %
    % and for a message at time t between keyframes k and k+1, C(t) is C_k and
    % C_{k+1} interpolated (slerp on the rotation, linear on the translation)
    % and applied as A_corrected = C(t) * A_raw(t).
    %
    % Messages outside [tKF(1), tKF(end)] are CLAMPED to the nearest keyframe's
    % correction rather than dropped -- they are the head and tail of the bag,
    % and dropping them would leave the first and last camera frames with no
    % pose at all. That is an extrapolation and the count is printed.

    nKF = numel(posesOpt);
    if nKF < 2
        warning('writeCorrectedOdometry:tooFewKeyframes', ...
            'Need at least 2 keyframes to interpolate the correction; skipping.');
        return;
    end

    % --- per-keyframe correction transform, as (quaternion, translation) ----
    qC = zeros(nKF, 4);   % [w x y z], MATLAB's rotm2quat order
    pC = zeros(nKF, 3);
    for k = 1:nKF
        Ck = posesOpt(k).A / posesKFRaw(k).A;   % == A_opt * inv(A_raw)
        qC(k, :) = rotm2quat(Ck(1:3, 1:3));
        pC(k, :) = Ck(1:3, 4)';
    end

    % --- apply to every raw pose -------------------------------------------
    n = numel(tOdom);
    posOut  = zeros(n, 3);
    quatOut = zeros(n, 4);   % [w x y z]
    nClamped = 0;
    for i = 1:n
        t = tOdom(i);
        if t <= tKF(1)
            k = 1;    u = 0;    nClamped = nClamped + 1;
        elseif t >= tKF(end)
            k = nKF-1; u = 1;   nClamped = nClamped + 1;
        else
            k = find(tKF <= t, 1, 'last');
            k = min(max(k, 1), nKF-1);
            u = (t - tKF(k)) / (tKF(k+1) - tKF(k));
        end
        qc = quatSlerpLocal(qC(k, :), qC(k+1, :), u);
        pc = (1-u) * pC(k, :) + u * pC(k+1, :);

        C = eye(4);
        C(1:3, 1:3) = quat2rotm(qc);
        C(1:3, 4)   = pc';

        A = C * posesRaw(i).A;
        posOut(i, :)  = A(1:3, 4)';
        quatOut(i, :) = rotm2quat(A(1:3, 1:3));
    end
    fprintf('\nCorrected odometry: %d message(s) over %.1f s', n, tOdom(end)-tOdom(1));
    if nClamped > 0
        fprintf(' (%d outside the keyframe span, correction clamped)', nClamped);
    end
    fprintf('\n');

    if ~exist(savedOdomDir, 'dir')
        mkdir(savedOdomDir);
    end

    % --- csv ---------------------------------------------------------------
    % Written first and unconditionally: it does not depend on the ROS Toolbox
    % writer being available, and it is the form a human can diff.
    csvFile = fullfile(savedOdomDir, [stamp '.csv']);
    fid = fopen(csvFile, 'w');
    fprintf(fid, 't_sec,x,y,z,qx,qy,qz,qw\n');
    for i = 1:n
        fprintf(fid, '%.9f,%.6f,%.6f,%.6f,%.9f,%.9f,%.9f,%.9f\n', ...
            tOdom(i), posOut(i,1), posOut(i,2), posOut(i,3), ...
            quatOut(i,2), quatOut(i,3), quatOut(i,4), quatOut(i,1));
    end
    fclose(fid);
    fprintf('  csv: %s\n', csvFile);

    % --- rosbag2 -----------------------------------------------------------
    % ros2bagwriter creates the folder itself and refuses an existing one.
    bagDir = fullfile(savedOdomDir, stamp);
    if exist(bagDir, 'dir')
        warning('writeCorrectedOdometry:bagExists', ...
            'Bag folder already exists, not overwriting:\n  %s', bagDir);
        return;
    end
    try
        writer = ros2bagwriter(bagDir);
        cleanupWriter = onCleanup(@() delete(writer));
        msg = ros2message('nav_msgs/Odometry');
        msg.header.frame_id = frameId;
        msg.child_frame_id  = childFrameId;
        for i = 1:n
            secs = floor(tOdom(i));
            nsec = round((tOdom(i) - secs) * 1e9);
            if nsec >= 1e9      % rounding can carry over into the next second
                secs = secs + 1;
                nsec = nsec - 1e9;
            end
            msg.header.stamp.sec     = int32(secs);
            msg.header.stamp.nanosec = uint32(nsec);
            msg.pose.pose.position.x = posOut(i, 1);
            msg.pose.pose.position.y = posOut(i, 2);
            msg.pose.pose.position.z = posOut(i, 3);
            msg.pose.pose.orientation.w = quatOut(i, 1);
            msg.pose.pose.orientation.x = quatOut(i, 2);
            msg.pose.pose.orientation.y = quatOut(i, 3);
            msg.pose.pose.orientation.z = quatOut(i, 4);
            write(writer, topicName, tOdom(i), msg);
        end
        clear cleanupWriter;   % flushes and closes the bag
        fprintf('  bag: %s  (topic %s)\n', bagDir, topicName);
        % ros2bagwriter leaves type_description_hash empty and the Python
        % reader (rosbags) asserts on it, so the bag does not open until the
        % digest is filled in. One command, once per bag, idempotent.
        fprintf(['  NOTE: run this once before reading the bag from Python:\n' ...
                 '    py livox_odometry_loader.py --bag "%s" --fix-digests\n'], bagDir);
    catch ME
        warning('writeCorrectedOdometry:bagFailed', ...
            ['Could not write the rosbag2 (%s). The .csv above still holds ' ...
             'every corrected pose.\n  %s'], ME.identifier, ME.message);
    end
end

function q = quatSlerpLocal(q0, q1, u)
    % Slerp between two [w x y z] quaternions. Written out rather than taken
    % from a toolbox so this file keeps the same dependencies it already had
    % (quat2rotm / rotm2quat only).
    q0 = q0 / norm(q0);
    q1 = q1 / norm(q1);
    d  = dot(q0, q1);
    if d < 0            % same rotation, opposite sign: take the short way round
        q1 = -q1;
        d  = -d;
    end
    if d > 0.9995       % almost parallel: slerp is ill-conditioned, lerp is exact enough
        q = q0 + u * (q1 - q0);
        q = q / norm(q);
        return;
    end
    th0 = acos(d);
    th  = th0 * u;
    q2  = q1 - q0 * d;
    q2  = q2 / norm(q2);
    q   = q0 * cos(th) + q2 * sin(th);
    q   = q / norm(q);
end
