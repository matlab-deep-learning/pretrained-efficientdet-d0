function iou = bboxOverlapIoU(groundTruthBoxes,boxes)
% Return IoU given ground truth boxes and boxes (anchors/box
% priors or region proposals).
%
% Output iou is a M-by-N matrix, where M is
% size(groundTruthBoxes,1) and N is size(boxes,1).     

% Copyright 2021 The MathWorks, Inc.

    if isempty(groundTruthBoxes)
        iou = zeros(0,size(boxes,1));
    elseif isempty(boxes)
        iou = zeros(size(groundTruthBoxes,1),0);
    else
        iou = bboxOverlapRatio(groundTruthBoxes, boxes, 'union');
    end
    
end