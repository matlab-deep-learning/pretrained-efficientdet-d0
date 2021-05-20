%% Code Generation for EfficientDet-d0
% The following code demonstrates code generation for a pre-trained 
% EfficientDet-D0 object detection network, trained on COCO 2017 dataset.

%% Setup
% Add path to the source directory.
addpath('src');

%% Download the Pre-trained Network
helper.downloadPretrainedEfficientDetD0;

%% Read and Preprocess Input Image.
% Read input image.
orgImage = imread('visionteam.jpg');

% Preprocess the image. 
[I,imageScale] = helper.preprocess(orgImage);

% Provide location of the mat file of the trained network.
matFile = 'model/efficientDetD0-coco.mat';

%% Run MEX code generation
% The efficientDetD0_predict.m is entry-point function that takes an input 
% image and give output from box and class network for different scales of
% feature map.  The function uses a persistent object efficientDetD0obj to 
% load the dlnetwork object and reuses the persistent object for prediction 
% on subsequent calls.
%
% To generate CUDA code for the efficientDetD0_predict entry-point function, 
% create a GPU code configuration object for a MEX target and set the 
% target language to C++. 
% 
% Use the coder.DeepLearningConfig (GPU Coder) function to create a CuDNN 
% deep learning configuration object and assign it to the DeepLearningConfig 
% property of the GPU code configuration object. 
% 
% Run the codegen command.
cfg = coder.gpuConfig('mex');
cfg.TargetLang = 'C++';
cfg.DeepLearningConfig = coder.DeepLearningConfig('cudnn');
args = {coder.Constant(matFile), I};
codegen -config cfg efficientDetD0_predict -args args -report

%% Run Generated MEX
% Call efficientDetD0_predict_mex on the input image.
out = efficientDetD0_predict_mex(matFile, I);

% Get classnames for COCO dataset.
classNames = helper.getCOCOClasess;

% Determine anchor boxes.
anchorBoxes = helper.generateAnchorBox;

% Apply postprocessing on the ouput.
[bboxes,scores,labels] = helper.postprocess(out, anchorBoxes, classNames, imageScale);

% Visualize results.
annotations = string(labels) + ": " + string(round(100*scores)) + "%";
img = insertObjectAnnotation(orgImage, 'rectangle', bboxes, cellstr(annotations));
figure, imshow(img);


% Copyright 2021 The MathWorks, Inc.