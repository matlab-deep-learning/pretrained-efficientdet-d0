function [assignments, positiveIndex, negativeIndex] = assignBoxes(...
        iou, posOverlap, negOverlap)
% Input iou has ground truth boxes along first dimension and
% boxes to assign along second dimension.
%
% Each box is assigned to exactly one ground truth.
% Each ground truth can be assigned to more than 1 box.

% Copyright 2021 The MathWorks, Inc.    

    assert(numel(posOverlap)==2);
    assert(numel(negOverlap)==2);
    
    % For each box, find best matching GT box (one with the largest
    % IoU score).
    [v,idx] = max(iou,[],1);
    
    % Assign each anchor a positive label if overlap threshold is
    % within specified range.
    lower = posOverlap(1);
    upper = posOverlap(2);
    positiveIndex =  reshape(v >= lower & v <= upper,[],1);
    
    % Generate a list of the ground truth box assigned to each
    % positive box.
    assignments = zeros(size(iou,2),1,'single');
    assignments(positiveIndex) = idx(positiveIndex);
    
    % Select regions to use as negative training samples
    lower = negOverlap(1);
    upper = negOverlap(2);
    negativeIndex =  reshape(v >= lower & v < upper,[],1);
    
    % Boxes marked as positives should not be negatives.
    negativeIndex(positiveIndex) = false;
end