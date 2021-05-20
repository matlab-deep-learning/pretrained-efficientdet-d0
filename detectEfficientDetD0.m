function [bboxes,scores,labels] = detectEfficientDetD0(dlnet, image, classNames, executionEnvironment)
% detectEfficientDetD0 runs prediction on a trained efficientdet-d0 network.
%
% Inputs:
% dlnet                - Pretrained efficientdet-d0 dlnetwork
% image                - RGB image to run prediction on. (H x W x 3)
% executionEnvironment - Environment to run predictions on. Specify cpu,
%                        gpu, or auto.
%
% Outputs:
% bboxes     - Final bounding box detections ([x y w h]) formatted as
%              NumDetections x 4.
% scores     - NumDetections x 1 classification scores.
% labels     - NumDetections x 1 categorical class labels.

% Copyright 2021 The MathWorks, Inc.

% Preprocess the image.
[newImage, imageScale] = helper.preprocess(image);

% Convert to dlarray.
XTest = dlarray(newImage, 'SSCB');

% If GPU is available, then convert data to gpuArray.
if (executionEnvironment == "auto" && canUseGPU) || executionEnvironment == "gpu"
    XTest = gpuArray(XTest);
end

% Output from EfficientDet model for the given test image.
out = cell(10,1);
[out{:}, ~] = predict(dlnet, XTest);

% Determine anchor boxes.
anchorBoxes = helper.generateAnchorBox;

% Postprocess the output.
[bboxes,scores,labels] = helper.postprocess(out, anchorBoxes, classNames, imageScale);
end