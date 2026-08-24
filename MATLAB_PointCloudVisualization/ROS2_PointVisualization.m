%% Point cloud visualization accumulated from a ROS2 bag
% Reads /cloud_registered from a ROS2 bag (.db3), accumulates multiple
% frames into a single cloud and displays it.
%
% /cloud_registered is FAST-LIO's already-registered output: the points
% are already expressed in the map frame, so consecutive frames can be
% concatenated directly without applying any transformation.

clear
close all
clc

%% 1. Parameters
bagPath    = "C:\Users\loren\Desktop\Dati_vfinal\SLAM\Lidar\rosbag2_2026_07_30-17_50_45\rosbag2_2026_07_30-17_50_45_0.db3";
topicName  = '/cloud_registered';

frameStep  = 5;      % 1 = all frames. Raise it (e.g. 5) if memory is not enough.
maxFrames  = Inf;    % limit on frames to read, Inf = no limit
voxelSize  = 0.05;   % m, voxel size for downsampling. 0 = disabled
markerSize = 20;     % point size on screen. pcshow's default is tiny

%% 2. Open bag
bag = ros2bagreader(bagPath);
disp('Topics available in the bag:');
disp(bag.AvailableTopics);

sel = select(bag, 'Topic', topicName);
nTotal = sel.NumMessages;
fprintf('\nTopic %s: %d messages\n', topicName, nTotal);

if nTotal == 0
    error(['No messages on %s.\n' ...
        'Check the topic name in the list above.'], topicName);
end

%% 3. Select which frames to read
idx = 1:frameStep:nTotal;
if numel(idx) > maxFrames
    idx = idx(1:maxFrames);
end
fprintf('Reading %d frames (step %d)\n', numel(idx), frameStep);

msgs = readMessages(sel, idx);

%% 4. Accumulate the frames into a single cloud
% Preallocate in a cell array and vertcat at the end: much faster than
% concatenating inside the loop, where the matrix would be reallocated
% on every iteration.
allXYZ = cell(numel(msgs), 1);
for i = 1:numel(msgs)
    allXYZ{i} = rosReadXYZ(msgs{i});
end
xyz = vertcat(allXYZ{:});

nRaw = size(xyz, 1);

% Remove invalid returns (NaN/Inf), present when the ray finds no
% surface within the sensor's range
xyz = xyz(all(isfinite(xyz), 2), :);
fprintf('Points read: %d, valid: %d (discarded %d)\n', ...
    nRaw, size(xyz,1), nRaw - size(xyz,1));

pc = pointCloud(xyz);

%% 5. Outlier filtering
% Two complementary filters, can be used together or separately.

% --- 5a. Geometric crop (ROI) ---
% Removes everything outside a box. This is the right filter for
% COHERENT spurious clusters (e.g. a line of points detached from the
% corridor), which statistical denoise does not remove because their
% points are close to each other and therefore do not come out as
% "isolated".
% Set useROI = false to disable it and see the whole cloud.
useROI = true;
roi = [-Inf Inf, ...    % X min max
       -Inf Inf, ...    % Y min max
       -10 10];     % Z min max, cuts above 4 m

if useROI
    inIdx  = findPointsInROI(pc, roi);
    nBefore = pc.Count;
    pc = select(pc, inIdx);
    fprintf('ROI crop: %d -> %d points (removed %d, %.1f%%)\n', ...
        nBefore, pc.Count, nBefore - pc.Count, 100*(nBefore - pc.Count)/nBefore);
end

% --- 5b. Statistical denoise ---
% Removes points whose mean distance to the k neighbours deviates by
% more than 'threshold' standard deviations from the global mean.
% Effective against diffuse scatter and single spurious returns.
% Raise threshold = more permissive filter, lower it = more aggressive.
useDenoise   = true;
denoiseK     = 20;    % number of neighbours considered
denoiseThres = 2;     % threshold in standard deviations

if useDenoise
    nBefore = pc.Count;
    pc = pcdenoise(pc, 'NumNeighbors', denoiseK, 'Threshold', denoiseThres);
    fprintf('Denoise: %d -> %d points (removed %d, %.1f%%)\n', ...
        nBefore, pc.Count, nBefore - pc.Count, 100*(nBefore - pc.Count)/nBefore);
end

%% 6. Optional downsampling
% Consecutive FAST-LIO frames overlap heavily, so most points are near
% duplicates. The voxel grid reduces the load without losing real
% coverage.
if voxelSize > 0
    pcView = pcdownsample(pc, 'gridAverage', voxelSize);
    fprintf('After %.0f cm downsampling: %d points (%.1f%% of original)\n', ...
        voxelSize*100, pcView.Count, 100*pcView.Count/pc.Count);
else
    pcView = pc;
end

%% 7. Cloud extent
fprintf('\nExtent [min max] in metres:\n');
fprintf('  X: %7.2f  %7.2f\n', pcView.XLimits);
fprintf('  Y: %7.2f  %7.2f\n', pcView.YLimits);
fprintf('  Z: %7.2f  %7.2f\n', pcView.ZLimits);

%% 8. Visualization
figure('Name', sprintf('%s - %d frames - %d points', ...
    topicName, numel(idx), pcView.Count), 'Color', 'k');

pcshow(pcView, 'MarkerSize', markerSize);

xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
title(sprintf('%d accumulated frames, %d points', numel(idx), pcView.Count), ...
    'Color', 'w');
axis equal;
grid on;
colormap(gca, turbo);   % color as a function of Z height
