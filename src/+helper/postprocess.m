function [selectedBboxes,selectedScores,selectedLabels] = postprocess(out, anchorBoxes, classNames, imageScale)
% This function postprocesses the output.

% Copyright 2021 The MathWorks, Inc.

numClasses = size(classNames,1);
numAnchorsPerLevel = 9;

bboxesOutAct = cellfun(@(x)reshape(permute(reshape(permute(stripdims(x),[3 2 1]),4,numAnchorsPerLevel,[]),[2 3 1]),[],4),...
    out(end:-2:2,:),'UniformOutput',false);
bboxesOutAct = cat(1,bboxesOutAct{:}); % [ty, tx, th, tw]
bboxesOutAct = extractdata(bboxesOutAct);

classes = cellfun(@(x)reshape(permute(reshape(permute(stripdims(x),[3 2 1]),numClasses,numAnchorsPerLevel,[]),[2 3 1]),[],numClasses),...
    out(end-1:-2:1,:),'UniformOutput',false);
classes = cat(1,classes{:});
classes = extractdata(classes);

% Extract the center, width and height of the bounding boxes.
bboxes_cyx = bboxesOutAct(:,1:2) .* anchorBoxes(:,3:4) + anchorBoxes(:,1:2);
bboxes_hw = exp(bboxesOutAct(:,3:4)) .* anchorBoxes(:,3:4);

% Convert diagonal pair of box corner [y1, x1, y2, x2] to top-left format [x1, y1, width, height].
bboxes = [bboxes_cyx(:,[2 1]) - bboxes_hw(:,[2 1])/2, bboxes_hw(:,[2 1])];
bboxes(:,1:2) = bboxes(:,1:2) + 1; % Convert zero origin to one origin.

[scores, classIndices] = max(classes,[],2);
scores = 1./(1+exp(-scores));

labels = categorical(classNames(classIndices));

% Select bounding boxes, scores and labels.
confidenceThreshold = 0.5;
ind = scores > confidenceThreshold;

overlapThreshold = 0.5;
[selectedBboxes,selectedScores,selectedLabels] = ...
    selectStrongestBboxMulticlass(bboxes(ind,:),scores(ind),labels(ind),...
    "OverlapThreshold",overlapThreshold);

% Remove 'background' category if present.
finalInd = selectedLabels == 'background';

selectedBboxes = selectedBboxes(~finalInd,:);
selectedScores = selectedScores(~finalInd);
selectedLabels = selectedLabels(~finalInd);
selectedLabels = removecats(selectedLabels, 'background');

% Scale the generated bounding boxes to size of the input image
selectedBboxes = [selectedBboxes(:,1:2)-1 selectedBboxes(:,3:4)]./imageScale + [1 1 0 0];

selectedBboxes = gather(selectedBboxes);
selectedScores = gather(selectedScores);
selectedLabels = gather(selectedLabels);
end