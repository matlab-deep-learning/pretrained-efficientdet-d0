%% Transfer Learning Using Pretrained EfficientDet-D0 Network
% The following code demonstrates how to perform transfer learning using the 
% pretrained EfficientDet-D0 network for object detection. This script uses 
% the "configureEfficientDetD0" function to create a custom EfficientDet-DO 
% network using the pretrained model.

%% Setup
% Add path to the source directory.
addpath('src');

%% Download Pretrained Model
model = helper.downloadPretrainedEfficientDetD0();
net = model.net;

%% Load Data
% This example uses a small labeled dataset that contains 295 images. Many 
% of these images come from the Caltech Cars 1999 and 2001 data sets, available 
% at the Caltech Computational Vision <http://www.vision.caltech.edu/archive.html 
% website>, created by Pietro Perona and used with permission. Each image contains 
% one or two labeled instances of a vehicle. A small data set is useful for exploring 
% the EfficientDet-D0 training procedure, but in practice, more labeled images 
% are needed to train a robust network.
% 
% Unzip the vehicle images and load the vehicle ground truth data. 
unzip vehicleDatasetImages.zip
data = load('vehicleDatasetGroundTruth.mat');
vehicleDataset = data.vehicleDataset;

% Add the full path to the local vehicle data folder.
vehicleDataset.imageFilename = fullfile(pwd, vehicleDataset.imageFilename);

% Note: In case of multiple classes, the data can also organized as three 
% columns where the first column contains the image file names with paths, the 
% second column contains the bounding boxes and the third column must be a cell 
% vector that contains the label names corresponding to each bounding box. For 
% more information on how to arrange the bounding boxes and labels, see 'boxLabelDatastore'.
% 
% All the bounding boxes must be in the form [x y width height]. This vector 
% specifies the upper left corner and the size of the bounding box in pixels.
% 
% Split the data set into a training set for training the network, and a test 
% set for evaluating the network. Use 60% of the data for training set and the 
% rest for the test set.
rng(0);
shuffledIndices = randperm(height(vehicleDataset));
idx = floor(0.6 * length(shuffledIndices));
trainingDataTbl = vehicleDataset(shuffledIndices(1:idx), :);
testDataTbl = vehicleDataset(shuffledIndices(idx+1:end), :);

% Create an image datastore for loading the images.
imdsTrain = imageDatastore(trainingDataTbl.imageFilename);
imdsTest = imageDatastore(testDataTbl.imageFilename);

% Create a datastore for the ground truth bounding boxes.
bldsTrain = boxLabelDatastore(trainingDataTbl(:, 2:end));
bldsTest = boxLabelDatastore(testDataTbl(:, 2:end));

% Combine the image and box label datastores.
trainingData = combine(imdsTrain, bldsTrain);
testData = combine(imdsTest, bldsTest);

% Use 'validateInputData' to detect invalid images, bounding boxes or labels 
% i.e., 
%  * Samples with invalid image format or containing NaNs
%  * Bounding boxes containing zeros/NaNs/Infs/empty
%  * Missing/non-categorical labels. 
%
% The values of the bounding boxes should be finite, positive, non-fractional, 
% non-NaN and should be within the image boundary with a positive height and width. 
% Any invalid samples must either be discarded or fixed for proper training.
helper.validateInputData(trainingData);
helper.validateInputData(testData);

%% Data Augmentation
% Data augmentation is used to improve network accuracy by randomly transforming 
% the original data during training. By using data augmentation, you can add more 
% variety to the training data without actually having to increase the number 
% of labeled training samples.
% 
% Use 'transform' function to apply custom data augmentations to the training 
% data. The 'augmentData' helper function, applies the following augmentations 
% to the input data.
%
% * Color jitter augmentation in HSV space
% * Random horizontal flip
% * Random scaling by 10 percent
augmentedTrainingData = transform(trainingData, @helper.augmentData);

% Read the same image four times and display the augmented training data.
% Visualize the augmented images.
augmentedData = cell(4,1);
for k = 1:4
    data = read(augmentedTrainingData);
    augmentedData{k} = insertShape(data{1,1}, 'Rectangle', data{1,2});
    reset(augmentedTrainingData);
end
figure
montage(augmentedData, 'BorderSize', 10)

%% Preprocess Training Data
% Preprocess the augmented training data to prepare for training. The preprocessTrainData 
% helper function, applies the following preprocessing operations to the input data.
%
% * Normalizes input image using vgg mean and standard deviation.
% * Resizes the larger dimension of the input image to size 512.
% * Applies zero padding to the resized image to make its resolution [512 512].
%   Input size of the pretrained network is [512 512 3]. Hence, all the input
%   images are preprocessed to have this size.
% * Rescales the bounding boxes as per the scale of resized image. 
preprocessedTrainingData = transform(augmentedTrainingData, @(data)helper.preprocessTrainData(data));
 
% Read the preprocessed training data.
data = read(preprocessedTrainingData);

% Display the image with the bounding boxes.
I = data{1,1};
bbox = data{1,2};
annotatedImage = insertShape(I, 'Rectangle', bbox);
annotatedImage = imresize(annotatedImage,2);
figure
imshow(annotatedImage)

% Reset the datastore.
reset(preprocessedTrainingData);

%% Modify Pretrained EfficientDet-D0 Network
% Specify the names of the object classes, number of object classes to detect, 
% and generate the anchor boxes using 'generateAnchorBox' function. Then, modify 
% the pretrained netowork's output layers using configureEfficientDetD0 function.

% Specify classnames.
classNames = {'vehicle'};
numClasses = size(classNames, 2);

% Add 'background' class to the existing class names.
classNames{end+1,1} = 'background';

% Determine anchor boxes.
anchorBoxes = helper.generateAnchorBox;

% Configure pretrained model for transfer learning.
[lgraph, networkOutputs] = configureEfficientDetD0(net, numClasses);

%% Specify Training Options
% Specify these training options.
%
% * Set the number of epochs to be 100.
% * Set the mini batch size as 8. Stable training can be possible with higher 
%   learning rates when higher mini batch size is used. Although, this should 
%   be set depending on the available memory.
% * Set the learning rate to 0.001.
% * Set the L2 regularization factor to 0.0005.
% * Specify the penalty threshold as 0.5. Detections that overlap less than 
%   0.5 with the ground truth are penalized.
% * Initialize the velocity as []. 
%   This is used by SGDM optimizer to store the velocity of gradients.
numEpochs = 100;
miniBatchSize = 8;
learningRate = 0.001;
warmupPeriod = 1000;
l2Regularization = 0.0005;
penaltyThreshold = 0.5;
velocity = [];

%% Train Model
% Train on a GPU, if one is available. Using a GPU requires Parallel Computing 
% Toolbox™ and a CUDA® enabled NVIDIA® GPU.
% 
% Use the 'minibatchqueue' function to split the preprocessed training data 
% into batches with function 'myMiniBatchFcn' which returns the 
% batched images, bounding boxes and the respective class labels. For 
% faster extraction of the batch data for training, 'dispatchInBackground' should 
% be set to "true" which ensures the usage of parallel pool.
% 
% 'minibatchqueue' automatically detects the availability of a GPU. If you do 
% not have a GPU, or do not want to use one for training, set the 'OutputEnvironment' 
% parameter to "cpu". 
if canUseParallelPool
   dispatchInBackground = true;
else
   dispatchInBackground = false;
end

if canUseGPU
    executionEnvironment = "gpu";
else
    executionEnvironment = "cpu";
end

myMiniBatchFcn = @(img, boxes, labels) deal(cat(4, img{:}), boxes, labels);

mbqTrain = minibatchqueue(preprocessedTrainingData, 3, "MiniBatchFormat", ["SSCB", "", ""],...
                            "MiniBatchSize", miniBatchSize,...
                            "OutputCast", ["single","",""],...
                            "OutputAsDlArray", [true, false, false],...
                            "DispatchInBackground", dispatchInBackground,...
                            "MiniBatchFcn", myMiniBatchFcn,...
                            "OutputEnvironment", [executionEnvironment,"cpu","cpu"]);

% To train the network with a custom training loop and enable automatic differentiation, 
% convert the layer graph to a dlnetwork object. Then create the training progress 
% plotter using helping function configureTrainingProgressPlotter. 
% 
% Finally, specify the custom training loop. For each iteration:
%
% * Read data from the 'minibatchqueue'. If it doesn't have any more data, reset 
%   the 'minibatchqueue' and shuffle. 
% * Evaluate the model gradients using 'dlfeval' and the 'modelGradients' function. 
%   The function 'modelGradients', listed as a supporting function, returns the 
%   gradients of the loss with respect to the learnable parameters in 'net', the 
%   corresponding mini-batch loss, and the state of the current batch.
% * Apply a weight decay factor to the gradients to regularization for more 
%   robust training.
% * Update the network parameters using the 'sgdmupdate' function.
% * Update the 'state' parameters of 'net' with the moving average.
% * Display the learning rate, total loss, and the individual losses for every 
%   iteration. These can be used to interpret how the respective losses are changing 
%   in each iteration. For example, a sudden spike in the box loss after few iterations 
%   implies that there are Inf or NaNs in the predictions.
% * Update the training progress plot.  
%
% The training can also be terminated if the loss has saturated for few epochs. 
rng('default');
modelName = "trainedNet";

start = tic;

% Convert layer graph to dlnetwork.
net = dlnetwork(lgraph);

% Create subplots for the learning rate and mini-batch loss.
fig = figure;
[lossPlotter, learningRatePlotter] = helper.configureTrainingProgressPlotter(fig);

iteration = 0;

% Custom training loop.
for epoch = 1:numEpochs
    reset(mbqTrain);
    shuffle(mbqTrain);
    
    while(hasdata(mbqTrain))
        iteration = iteration + 1;
        
        [imgTrain, bboxTrain, labelTrain] = next(mbqTrain);
        
        % Evaluate the model gradients and loss using dlfeval and the modelGradients function.
        [gradients, state, lossInfo] = dlfeval(@modelGradients, net, imgTrain, bboxTrain, labelTrain, anchorBoxes, penaltyThreshold, classNames, networkOutputs);
        
        % Apply L2 regularization.
        gradients = dlupdate(@(g,w) g + l2Regularization*w, gradients, net.Learnables);
        
        % Determine the current learning rate value.
        currentLR = helper.piecewiseLearningRateWithWarmup(iteration, epoch, learningRate, warmupPeriod, numEpochs);
       
        % Update the network learnable parameters using the SGDM optimizer.
        [net, velocity] = sgdmupdate(net, gradients, velocity, currentLR);
        
        % Update the state parameters of dlnetwork.
        net.State = state;
        
        % Display progress.
        if mod(iteration,10) == 1
            helper.displayLossInfo(epoch, iteration, currentLR, lossInfo);
        end
        
        % Update training plot with new points.
        helper.updatePlots(lossPlotter, learningRatePlotter, iteration, currentLR, lossInfo.totalLoss);
    end
end

save(modelName,"net");

%% Evaluate Model
% Computer Vision System Toolbox™ provides object detector evaluation functions 
% to measure common metrics such as average precision (evaluateDetectionPrecision) 
% and log-average miss rates (evaluateDetectionMissRate). In this example, the 
% average precision metric is used. The average precision provides a single number 
% that incorporates the ability of the detector to make correct classifications 
% (precision) and the ability of the detector to find all relevant objects (recall).
% 
% Following these steps to evaluate the trained dlnetwork object net on 
% test data.
%
% * Specify the confidence threshold as 0.5 to keep only detections with confidence 
%   scores above this value.
% * Specify the overlap threshold as 0.5 to remove overlapping detections.
% * Collect the detection results by running the detector on test data. 
%   Use the supporting function detectEfficientDetD0 to get the bounding boxes and 
%   class labels.
% * Call evaluateDetectionPrecision with predicted results and test data as arguments. 

% Specify confidence threshold and overlap threshold.
confidenceThreshold = 0.5;
overlapThreshold = 0.5;

% Run detector on images in the test set and collect results.
results = table('Size', [0 3], ...
    'VariableTypes', {'cell','cell','cell'}, ...
    'VariableNames', {'Boxes','Scores','Labels'});

T = table('Size', [0 1], ...
    'VariableTypes', {'cell'}, ...
    'VariableNames', {'vehicle'});

reset(testData)
while hasdata(testData)
    % Read the datastore.
    data = read(testData);
    image = data{1};
    bbox = data{2};
    label = data{3};
    
    % Generate detection results.
    executionEnvironment = 'auto';
    [bboxes,scores,labels] = detectEfficientDetD0(net, image, classNames, executionEnvironment);
    
    % Collect the results.
    tbl = table({bboxes}, {scores}, {labels}, 'VariableNames', {'Boxes','Scores','Labels'});
    newrow = {};
    for i2 = 1:numel(classNames(1:end-1))
        aa = label == classNames{i2};
        bb = bbox(aa,:);
        newrow{end+1} = bb;
    end
    T = [T;newrow];
    results = [results; tbl];
end

% Evaluate the object detector using Average Precision metric.
[ap, recall, precision] = evaluateDetectionPrecision(results, T);

% The precision-recall (PR) curve shows how precise a detector is at varying 
% levels of recall. Ideally, the precision is 1 at all recall levels.
figure
for k = 1:numel(recall)
    plot(recall, precision)
    hold on;
end
xlabel('Recall')
ylabel('Precision')
legend(classNames);
xlim([0 1]);
ylim([0 1]);
grid on
title(sprintf('mAP_{%d} = %2.1f%%',int32(overlapThreshold*100), mean(ap)*100))

%% Detect Objects Using EfficientDet-D0
% Use the network for object detection. 
% * Read an image.
% * Convert the image to a dlarray and use a GPU if one is available..
% * Use the supporting function efficientdetDetect to get the predicted bounding 
%   boxes, confidence scores, and class labels. 
% * Display the image with bounding boxes and confidence scores.

% Read the datastore.
reset(testData)
data = read(testData);

% Get the image.
img = data{1};

% Perform detection using trained EfficientDet-D0 network.
executionEnvironment = 'auto';
[bboxes,scores,labels] = detectEfficientDetD0(net, img, classNames, executionEnvironment);

% Display result.
if ~isempty(scores)
    Iout = insertObjectAnnotation(img, 'rectangle', gather(bboxes), gather(scores));
else
    Iout = im2uint8(I);
end
figure
imshow(Iout)


%% References
% [1] Mingxing Tan, Ruoming Pang, Quoc V. Le, “EfficientDet: Scalable and Efficient 
% Object Detection.” Proceedings of the IEEE Conference on Computer Vision and 
% Pattern Recognition (2020),  
% 
% 
% Copyright 2021 The MathWorks, Inc.