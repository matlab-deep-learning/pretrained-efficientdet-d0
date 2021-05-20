%% Object Detection Using Pretrained EfficientDet-D0 Network
% The following code demonstrates running object detection using a pretrained 
% EfficientDet-D0 network, trained on COCO 2017 dataset.

%% Prerequisites
% To run this example you need the following prerequisites - 
%
% # MATLAB (R2021a or later) with Computer Vision and Deep Learning Toolbox.
% # Pretrained EfficientDet-D0 network (download instructions below).

%% Setup
% Add path to the source directory.
addpath('src');

%% Download the Pre-trained Network
model = helper.downloadPretrainedEfficientDetD0;
net = model.net;

%% Perform Object Detection Using EfficientDet-D0 Network
% Inference is performed on the pretrained EfficientDet-D0 network. 
% Read test image.
img = imread('visionteam.jpg');

% Get classnames for COCO dataset.
classNames = helper.getCOCOClasess;

% Perform detection using pretrained model.
executionEnvironment = 'auto';
[bboxes,scores,labels] = detectEfficientDetD0(net, img, classNames, executionEnvironment);

% Visualize results.
annotations = string(labels) + ": " + string(round(100*scores)) + "%";
img = insertObjectAnnotation(img, 'rectangle', bboxes, cellstr(annotations));
figure, imshow(img);
 

% Copyright 2021 The MathWorks, Inc.