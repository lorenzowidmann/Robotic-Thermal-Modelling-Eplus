%% Point cloud visualization from a fast_lio_sam_sc_qn result bag (ROS1)
% Reads /keyframe_pcd and /keyframe_pose from a ROS1 bag produced by
% fast_lio_sam_sc_qn (SC-QN), rebuilds the corrected map from the raw
% per-keyframe scans, and displays it.
%
% Unlike /cloud_registered, /keyframe_pcd is NOT already in the map frame:
% each keyframe's cloud is stored in the LOCAL sensor frame at capture
% time, so it must be rotated/translated by its own /keyframe_pose (the
% pose graph's CORRECTED pose, i.e. after loop closure) before merging.
% No voxelization is applied here, so this is the raw point density
% actually captured, higher than result.pcd (which is voxelized on save).

bagFile = "C:\Users\loren\Desktop\Measurment_v2\ClaudeCode\FAST-LIO-SAM-SC-QN\fast_lio_sam_sc_qn\result.bag";
bag = rosbag(bagFile);

poseMsgs = readMessages(select(bag, "Topic", "/keyframe_pose"), "DataFormat", "struct");
pcdMsgs  = readMessages(select(bag, "Topic", "/keyframe_pcd"),  "DataFormat", "struct");

n = numel(poseMsgs);
clouds = cell(1, n);

for i = 1:n
    p = poseMsgs{i}.Pose.Position;
    q = poseMsgs{i}.Pose.Orientation;        % ROS: x y z w
    R = quat2rotm([q.W q.X q.Y q.Z]);        % MATLAB quat2rotm wants [w x y z]
    T = [p.X p.Y p.Z];

    xyzLocal = rosReadXYZ(pcdMsgs{i});
    xyzWorld = (R * xyzLocal')' + T;
    clouds{i} = pointCloud(xyzWorld);
end

mergedMap = pccat([clouds{:}]);
fprintf('Raw points (no voxel): %d\n', mergedMap.Count);

%% ROI crop
% Removes everything outside a box. This is the right filter for
% COHERENT spurious clusters (e.g. a line of points detached from the
% corridor), which statistical denoise does not remove because their
% points are close to each other and therefore do not come out as
% "isolated".
% Set useROI = false to disable it and see the whole cloud.
useROI = true;
roi = [12 Inf, ...    % X min max
       -2 2, ...      % Y min max
       -Inf Inf];     % Z min max

if useROI
    inIdx  = findPointsInROI(mergedMap, roi);
    nBefore = mergedMap.Count;
    mergedMap = select(mergedMap, inIdx);
    fprintf('ROI crop: %d -> %d points (removed %d, %.1f%%)\n', ...
        nBefore, mergedMap.Count, nBefore - mergedMap.Count, 100*(nBefore - mergedMap.Count)/nBefore);
end

%% Statistical denoise
% Removes points whose mean distance to the k neighbours deviates by
% more than 'threshold' standard deviations from the global mean.
% Effective against diffuse scatter and single spurious returns.
% Raise threshold = more permissive filter, lower it = more aggressive.
useDenoise   = true;
denoiseK     = 20;    % number of neighbours considered
denoiseThres = 3;     % threshold in standard deviations

if useDenoise
    nBefore = mergedMap.Count;
    mergedMap = pcdenoise(mergedMap, 'NumNeighbors', denoiseK, 'Threshold', denoiseThres);
    fprintf('Denoise: %d -> %d points (removed %d, %.1f%%)\n', ...
        nBefore, mergedMap.Count, nBefore - mergedMap.Count, 100*(nBefore - mergedMap.Count)/nBefore);
end

figure('Color','k');
pcshow(mergedMap, 'MarkerSize', 10);
axis equal; grid on; colormap(gca, turbo);
title('SC-QN — raw keyframes, no voxelization', 'Color', 'w');
