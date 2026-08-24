%% Point cloud visualization accumulated from a ROS2 bag -- no filters
% Reads /cloud_registered from a ROS2 bag (.db3), accumulates multiple
% frames into a single cloud and displays it. Same as ROS2_PointVisualization.m
% but with ROI crop, statistical denoise and voxel downsampling removed --
% just the raw accumulated cloud.
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

frameStep  = 1;      % 1 = all frames. Raise it (e.g. 5) if memory is not enough.
maxFrames  = Inf;    % limit on frames to read, Inf = no limit
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

%% 5. Cloud extent
fprintf('\nExtent [min max] in metres:\n');
fprintf('  X: %7.2f  %7.2f\n', pc.XLimits);
fprintf('  Y: %7.2f  %7.2f\n', pc.YLimits);
fprintf('  Z: %7.2f  %7.2f\n', pc.ZLimits);

%% 6. Visualization
figure('Name', sprintf('%s - %d frames - %d points', ...
    topicName, numel(idx), pc.Count), 'Color', 'k');

pcshow(pc, 'MarkerSize', markerSize);

xlabel('X (m)'); ylabel('Y (m)'); zlabel('Z (m)');
title(sprintf('%d accumulated frames, %d points (no filters)', numel(idx), pc.Count), ...
    'Color', 'w');
axis equal;
grid on;
colormap(gca, turbo);   % color as a function of Z height
