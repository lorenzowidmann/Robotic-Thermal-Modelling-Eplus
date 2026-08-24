%% ZED 2i Intrinsic Calibration with Outlier Identification
% Loads checkerboard images, runs calibration, reports which images failed
% detection, and ranks the rest by reprojection error so the worst can be
% moved out manually before rerunning.
clear
clc
close all

%% 1. Load images (png, non-recursive by default)
folderPath = "C:\Users\loren\Desktop\Dati_vfinal\Calibration\ZedInt\SmallCheck1080p - 25 mm";
imageFiles = dir(fullfile(folderPath, '*.png'));
imagePaths = fullfile({imageFiles.folder}, {imageFiles.name});
fprintf('Found %d images in %s\n', numel(imagePaths), folderPath);

%% 2. Detect checkerboard corners
[imagePoints, boardSize, imagesUsed] = detectCheckerboardPoints(imagePaths);

% imagesUsed is a logical array the same length as imagePaths, telling you
% which images actually had a successfully detected board.
validPaths   = imagePaths(imagesUsed);
droppedPaths = imagePaths(~imagesUsed);

fprintf('Board detected in %d / %d images  (boardSize = [%d,%d] squares, %d corners each)\n', ...
    numel(validPaths), numel(imagePaths), ...
    boardSize(1), boardSize(2), prod(boardSize - 1));

%% 2b. Dropped images report (filenames + visual grid)
fprintf('\n=== DROPPED IMAGES REPORT ===\n');
fprintf('Total images:    %d\n', numel(imagePaths));
fprintf('Detected:        %d\n', numel(validPaths));
fprintf('Dropped:         %d\n', numel(droppedPaths));

if ~isempty(droppedPaths)
    fprintf('\n--- Filenames where NO checkerboard was detected ---\n');
    for i = 1:numel(droppedPaths)
        [~, name, ext] = fileparts(droppedPaths{i});
        fprintf('  %2d. %s%s\n', i, name, ext);
    end

    % Visual grid of the dropped images, so you can see WHY each one failed
    figure('Name', 'Dropped Images (no board detected)');
    nCols = ceil(sqrt(numel(droppedPaths)));
    nRows = ceil(numel(droppedPaths) / nCols);
    for i = 1:numel(droppedPaths)
        subplot(nRows, nCols, i);
        imshow(imread(droppedPaths{i}));
        [~, name] = fileparts(droppedPaths{i});
        title(name, 'Interpreter', 'none', 'FontSize', 8);
    end
else
    fprintf('\nAll images had a successful board detection.\n');
end

%% 3. Generate world points and calibrate
squareSize = 25; % mm - measure the ACTUAL printed board, not the nominal PDF value
worldPoints = generateCheckerboardPoints(boardSize, squareSize);

imgSize = size(imread(validPaths{1}), [1 2]);   % [height width]
imgH = imgSize(1);
imgW = imgSize(2);
fprintf('\nImage size detected: %d x %d px (W x H)\n', imgW, imgH);

% Primary model: skew forced to zero. A ZED sensor has an orthogonal pixel
% grid, so zero skew is the physically correct assumption and MATLAB's default.
cameraParams = estimateCameraParameters(imagePoints, worldPoints, ...
    'ImageSize', imgSize, ...
    'EstimateTangentialDistortion', true, ...
    'NumRadialDistortionCoefficients', 2, ...
    'EstimateSkew', false);

% Secondary model: skew estimated, as a sanity check only. Skew trades off
% against fx and cx during the fit, so large shifts between the two models
% mean those parameters are weakly constrained by the data.
cameraParamsSkew = estimateCameraParameters(imagePoints, worldPoints, ...
    'ImageSize', imgSize, ...
    'EstimateTangentialDistortion', true, ...
    'NumRadialDistortionCoefficients', 2, ...
    'EstimateSkew', true);

%% 4. Show standard reprojection error plot
figure;
showReprojectionErrors(cameraParams);

%% 5. Per-image mean error WITH correct filename mapping
% NOTE: MATLAB supports PARTIAL checkerboard detection - corners that were not
% found come back as NaN in imagePoints, and therefore as NaN in
% ReprojectionErrors. A plain mean() would return NaN for any such image and
% poison the overall mean, so every aggregation below uses 'omitnan'.
errors = cameraParams.ReprojectionErrors;        % [numPoints x 2 x numImages]
perPointErr = sqrt(sum(errors.^2, 2));           % [numPoints x 1 x numImages]
meanErrorPerImage = squeeze(mean(perPointErr, 1, 'omitnan'));

% How complete was each board? (fraction of expected corners actually found)
nExpected = prod(boardSize - 1);
nFound = squeeze(sum(~isnan(imagePoints(:,1,:)), 1));
partialIdx = find(nFound < nExpected);

if ~isempty(partialIdx)
    fprintf('\n--- Images with PARTIAL board detection ---\n');
    fprintf('(expected %d corners each)\n', nExpected);
    for k = 1:numel(partialIdx)
        i = partialIdx(k);
        [~, name, ext] = fileparts(validPaths{i});
        fprintf('  %s%s  -> %d/%d corners (%.0f%%)\n', ...
            name, ext, nFound(i), nExpected, 100 * nFound(i) / nExpected);
    end
    fprintf(['Partial boards constrain the model less than complete ones.\n' ...
        'If a board is only fractionally detected, consider excluding it.\n']);
end

[sortedErr, idx] = sort(meanErrorPerImage, 'descend');

fprintf('\n--- Per-image mean reprojection error (worst to best) ---\n');
for i = 1:numel(idx)
    [~, name, ext] = fileparts(validPaths{idx(i)});
    fprintf('%2d. [err=%.4f px]  %s%s  (%d/%d corners)\n', ...
        i, sortedErr(i), name, ext, nFound(idx(i)), nExpected);
end

fprintf('\n--- Suggested outliers (error > overall mean + 1 std) ---\n');
threshold  = mean(meanErrorPerImage, 'omitnan') + std(meanErrorPerImage, 'omitnan');
outlierIdx = find(meanErrorPerImage > threshold);
if isempty(outlierIdx)
    fprintf('  none\n');
else
    for i = 1:numel(outlierIdx)
        [~, name, ext] = fileparts(validPaths{outlierIdx(i)});
        fprintf('  %s%s  (err=%.4f px)\n', name, ext, meanErrorPerImage(outlierIdx(i)));
    end
end

%% 6. Print final intrinsics
% Use the modern K property when available (already in standard orientation);
% fall back to transposing IntrinsicMatrix on older releases.
if isprop(cameraParams, 'K')
    K     = cameraParams.K;
    KSkew = cameraParamsSkew.K;
else
    K     = cameraParams.IntrinsicMatrix';
    KSkew = cameraParamsSkew.IntrinsicMatrix';
end

fx = K(1,1); fy = K(2,2); cx = K(1,3); cy = K(2,3);

fprintf('\n--- Intrinsics (skew = 0, model to report) ---\n');
fprintf('fx = %.4f\n', fx);
fprintf('fy = %.4f\n', fy);
fprintf('cx = %.4f\n', cx);
fprintf('cy = %.4f\n', cy);

fprintf('\nK =\n');
fprintf('  [ %10.4f  %10.4f  %10.4f ]\n', K(1,1), K(1,2), K(1,3));
fprintf('  [ %10.4f  %10.4f  %10.4f ]\n', K(2,1), K(2,2), K(2,3));
fprintf('  [ %10.4f  %10.4f  %10.4f ]\n', K(3,1), K(3,2), K(3,3));

rd = cameraParams.RadialDistortion;
td = cameraParams.TangentialDistortion;
fprintf('\nRadialDistortion     k1 = %+.6e   k2 = %+.6e\n', rd(1), rd(2));
fprintf('TangentialDistortion p1 = %+.6e   p2 = %+.6e\n', td(1), td(2));
fprintf('Images used: %d\n', numel(validPaths));
fprintf('Overall Mean Reprojection Error: %.4f px\n', mean(meanErrorPerImage, 'omitnan'));
fprintf('  (MATLAB''s own MeanReprojectionError: %.4f px)\n', cameraParams.MeanReprojectionError);

%% 6b. Principal point sanity check
% For a well constrained fit cx,cy should sit near the image centre. A large
% offset usually means the board never reached the edges of the frame.
fprintf('\n--- Principal point vs image centre ---\n');
fprintf('cx = %.2f   (centre %.1f, offset %+.2f px = %+.2f%% of width)\n', ...
    cx, imgW/2, cx - imgW/2, 100*(cx - imgW/2)/imgW);
fprintf('cy = %.2f   (centre %.1f, offset %+.2f px = %+.2f%% of height)\n', ...
    cy, imgH/2, cy - imgH/2, 100*(cy - imgH/2)/imgH);

%% 6c. FOV cross-check against the datasheet
% Independent validation: the focal lengths imply a field of view that can be
% compared against the ZED 2i spec, which the calibration never saw.
%   FOV_h = 2*atan(W / (2*fx))      FOV_v = 2*atan(H / (2*fy))
fovH = 2 * atand(imgW / (2 * fx));
fovV = 2 * atand(imgH / (2 * fy));
fprintf('\n--- FOV implied by these intrinsics ---\n');
fprintf('FOV horizontal = %.2f deg\n', fovH);
fprintf('FOV vertical   = %.2f deg\n', fovV);
fprintf('Compare against the ZED 2i datasheet FOV for this resolution mode.\n');

%% 6d. Skew vs no-skew comparison (sanity check only)
fprintf('\n--- No-skew (reported) vs skew-estimated (check) ---\n');
fprintf('%-8s %12s %12s %10s\n', 'param', 'skew off', 'skew on', 'delta');
fprintf('%-8s %12.4f %12.4f %10.4f\n', 'fx',   K(1,1), KSkew(1,1), KSkew(1,1) - K(1,1));
fprintf('%-8s %12.4f %12.4f %10.4f\n', 'fy',   K(2,2), KSkew(2,2), KSkew(2,2) - K(2,2));
fprintf('%-8s %12.4f %12.4f %10.4f\n', 'cx',   K(1,3), KSkew(1,3), KSkew(1,3) - K(1,3));
fprintf('%-8s %12.4f %12.4f %10.4f\n', 'cy',   K(2,3), KSkew(2,3), KSkew(2,3) - K(2,3));
fprintf('%-8s %12.4f %12.4f %10.4f\n', 'skew', K(1,2), KSkew(1,2), KSkew(1,2) - K(1,2));
fprintf('%-8s %12.4f %12.4f %10.4f\n', 'err',  ...
    cameraParams.MeanReprojectionError, cameraParamsSkew.MeanReprojectionError, ...
    cameraParamsSkew.MeanReprojectionError - cameraParams.MeanReprojectionError);

if KSkew(1,1) ~= 0
    fprintf('  -> axis angle theta = %.4f deg (90 = perfectly orthogonal)\n', ...
        acotd(KSkew(1,2) / KSkew(1,1)));
end
fprintf(['Adding skew always lowers reprojection error slightly (one extra free\n' ...
    'parameter), so a small improvement is not evidence the skew is real.\n']);

%% 7. Corner-spread diagnostic
% Where in the frame do the detected corners actually fall? Coefficients are
% only truly constrained where corners were observed; empty regions near the
% edges mean k1/k2 are extrapolated there rather than measured.
figure; hold on;
for i = 1:size(imagePoints, 3)
    plot(imagePoints(:,1,i), imagePoints(:,2,i), '.');
end
rectangle('Position', [0, 0, imgW, imgH], 'EdgeColor', 'k', 'LineWidth', 1.5);
title(sprintf('Corner locations across all calibration images (%d x %d)', imgW, imgH));
xlabel('u (px)'); ylabel('v (px)');
xlim([0 imgW]); ylim([0 imgH]);
axis ij; axis equal;
grid on;

% Rough coverage figure: fraction of the frame area spanned by the corners
uAll = imagePoints(:,1,:); uAll = uAll(~isnan(uAll));
vAll = imagePoints(:,2,:); vAll = vAll(~isnan(vAll));
covFrac = ((max(uAll) - min(uAll)) * (max(vAll) - min(vAll))) / (imgW * imgH);
fprintf('\nCorner bounding box: u [%.0f %.0f], v [%.0f %.0f]\n', ...
    min(uAll), max(uAll), min(vAll), max(vAll));
fprintf('Spans %.0f%% of the frame area. Low coverage means k1/k2 and cx/cy\n', 100*covFrac);
fprintf('are extrapolated near the edges rather than measured.\n');