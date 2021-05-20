function labels = boxLabelsFromAssignment(...
        assignments, groundTruthBoxLabels, ...
        positiveIndex, negativeIndex, ...
        classNames, backgroundClassName)            
% Return categorical vector of class names assigned to each box.
 
% Copyright 2021 The MathWorks, Inc. 
    
    % Preallocate categorical output.
    labels = repmat({''}, size(assignments,1), 1);
    labels = categorical(labels,[reshape(classNames,[],1); backgroundClassName]);

    labels(positiveIndex,:) = groundTruthBoxLabels(assignments(positiveIndex),:);
    labels(negativeIndex,:) = {backgroundClassName};   
end