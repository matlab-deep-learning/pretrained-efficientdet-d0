function [gradients, state, info] = modelGradients(net, XTrain, BTrain, LTrain, anchorBoxes, threshold, classNames, networkOutputs)
% The function modelGradients takes as input the dlnetwvalidateInputDataork object net, 
% a mini-batch of input data XTrain with corresponding ground truth boxes 
% BTrain and ground truth labels LTrain, anchor boxes, the specified penalty 
% threshold, and the network output names as input arguments and returns 
% the gradients of the loss with respect to the learnable parameters in net, 
% the corresponding mini-batch loss, and the state of the current batch. 
% 
% The model gradients function computes the total loss and gradients by performing 
% these operations.
%
% * Generate predictions from the input batch of images.
% * Collect predictions on the CPU for postprocessing.
% * Generate targets for loss computation by using the converted predictions 
%   and ground truth data. These targets are generated for bounding box positions 
%   (x, y, width, height) and class probabilities. 
% * Calculates the huber loss of the predicted bounding box coordinates with 
%   target boxes.
% * Determines the focal cross-entropy of the predicted class of object with 
%   the target.
% * Computes the total loss as the weighted sum of all losses.
% * Computes the gradients of learnables with respect to the total loss.

% Copyright 2021 The MathWorks, Inc.   

% Threshold ranges for deciding positive and negative anchors.
PositiveOverlapRange = [threshold 1];
NegativeOverlapRange = [0 threshold];

% Specify the background label.
BackgroundLabel = 'background';

% Extract the predictions from the network.
YPredictions = cell(size(networkOutputs));
[YPredictions{:}, state] = forward(net, XTrain);
YPredCell = YPredictions;

numAnchorsPerLevel = 9;
numclasses = size(classNames,1);
[~,~,~,n]=size(YPredCell{1,1});

bboxesOutAct = cellfun(@(x)reshape(permute(reshape(permute(stripdims(x),[3 2 1 4]),4,numAnchorsPerLevel,[],n),[2 3 1 4]),[],4,n),...
    YPredCell(end:-2:2,:),'UniformOutput',false);
bboxesOutAct = cat(1,bboxesOutAct{:}); % [ty, tx, th, tw]
bboxesOutAct = reshape(bboxesOutAct, [49104,1,4,n]);
predicted_locs = dlarray(bboxesOutAct, 'SSCB');

classOutAct = cellfun(@(x)reshape(permute(reshape(permute(stripdims(x),[3 2 1 4]),numclasses,numAnchorsPerLevel,[],n),[2 3 1 4]),[],numclasses,n),...
    YPredCell(end-1:-2:1,:),'UniformOutput',false);
classOutAct = cat(1,classOutAct{:});
classOutAct = reshape(classOutAct, [49104,1,numclasses,n]);
predicted_scores = dlarray(classOutAct, 'SSCB');

predicted_locs =stripdims(predicted_locs);
predicted_scores =stripdims(predicted_scores);

% Preparing target classes and target locations to compute losses.
batch_size = size(predicted_locs,4);
n_priors = 49104;
n_classes = size(predicted_scores,3);

true_location = zeros(n_priors,4,batch_size);

true_classes = [];
positive  = false(n_priors,batch_size);
negative = false(n_priors,batch_size);

for i = 1: batch_size
    boxes = BTrain{i};
    labels = LTrain{i};
    ClassNames = categories(LTrain{i});
    
    n_objects = size(boxes,1);
    
    groundTruthBoxes = [];
    groundTruthClasses = [];
    thisGroundTruthBBoxes = boxes;
    thisGroundTruthLabels = labels;
    for bidx = 1:n_objects
        if all(thisGroundTruthBBoxes(bidx, :))
            groundTruthBoxes = [groundTruthBoxes; thisGroundTruthBBoxes(bidx, :)]; %#ok<AGROW>
            groundTruthClasses = [groundTruthClasses; thisGroundTruthLabels(bidx)]; %#ok<AGROW>
        end
    end

    anchorBoxesTransformed = [anchorBoxes(:,2)-anchorBoxes(:,4)/2, anchorBoxes(:,1)-anchorBoxes(:,3)/2, anchorBoxes(:,4), anchorBoxes(:,3)];
    
    % Assign region proposals to ground truth boxes.
    [thisAssignment, positiveIndex, negativeIndex] = helper.assignBoxesToGroundTruthBoxes(...
        anchorBoxesTransformed, groundTruthBoxes, ...
        PositiveOverlapRange, ...
        NegativeOverlapRange);

     % Assign label to each region proposal.
    thisLabels = helper.boxLabelsFromAssignment(...
        thisAssignment, groundTruthClasses, positiveIndex, negativeIndex, ...
        ClassNames, BackgroundLabel);  
            
    % Create an array that maps ground truth box to positive
    % proposal box. i.e. this is the closest grouth truth box to
    % each positive region proposal.
    numBoxes = size(anchorBoxes, 1);
    targets = zeros(numBoxes , 4, 'like', groundTruthBoxes);
    for idx = 1:numBoxes
        if thisAssignment(idx) > 0
            G = groundTruthBoxes(thisAssignment(idx), :);
            P = anchorBoxesTransformed(idx, :);

            % positive sample regression targets
            targets(idx, :) = helper.targetEncode(G, P);
        end
    end
    true_classes = [true_classes,thisLabels];
    true_location(:,:,i) = targets;
    positive(:,i) = positiveIndex;
    negative(:,i) = negativeIndex;
end         
            
positive_priors = positive;
positive_priors_repmat = repmat(positive_priors,1,1,4);
positive_priors_repmat_1 = permute(positive_priors_repmat,[3 1 2 4]);
predicted_locsp = permute(squeeze(predicted_locs),[2 1 3 4]);
predicted_location = predicted_locsp(positive_priors_repmat_1);
predicted_locationf = dlarray(reshape(predicted_location,4,[]),'CB');

true_locsp = permute(squeeze(true_location),[2 1 3 4]);
true_locsf = true_locsp(positive_priors_repmat_1);
true_locsff = reshape(true_locsf,4,[]);  

predicted_scoresp = permute(squeeze(predicted_scores),[2,1,3]);
predicted_scoresf = dlarray(reshape(predicted_scoresp,n_classes,[]),'CB');
predicted_scoresf = softmax(predicted_scoresf);
true_classes_multi_batch = reshape(true_classes,1,[]);
classificationTargets = onehotencode(true_classes_multi_batch,1);
classificationTargets(isnan(classificationTargets)) = 0;

% Compute bounding box regression loss.
loc_Loss = huber(predicted_locationf,true_locsff,'TransitionPoint', 1/9, 'NormalizationFactor','none');
loc_Loss = 9.*loc_Loss;

% Compute class confidence loss.
conf_loss_all = sum(focalCrossEntropy(predicted_scoresf, classificationTargets,"Reduction",'none',"TargetCategories",'independent'));
conf_loss_all = reshape(conf_loss_all,n_priors,batch_size);  
conf_loss_pos = conf_loss_all(positive_priors);
conf_loss_neg = conf_loss_all(negative);

% Averaging the loss over only positive anchors.
n_positives = sum(positive_priors,1);
conf_loss = sum(conf_loss_pos) + sum(conf_loss_neg);
loc_Loss = sum(loc_Loss);
totalLoss = (50*loc_Loss + conf_loss)./sum(n_positives);

info.loc_Loss = loc_Loss;
info.conf_loss = conf_loss;
info.totalLoss = totalLoss;

% Compute gradients of learnables with regard to loss.
gradients = dlgradient(totalLoss, net.Learnables);
end