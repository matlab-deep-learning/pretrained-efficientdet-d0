function [assignments, positiveIndex, negativeIndex] = assignBoxesToGroundTruthBoxes(...
        boxes, groundTruthBoxes, posOverlap, negOverlap)
% Compute the Intersection-over-Union (IoU) metric between the ground truth 
% boxes and the region proposal boxes.

% Copyright 2021 The MathWorks, Inc.

    iou = helper.bboxOverlapIoU(groundTruthBoxes, boxes);
    
    % Assign based on IoU metric.
    [assignments, positiveIndex, negativeIndex] = ...
        helper.assignBoxes(iou, posOverlap, negOverlap);
    
end