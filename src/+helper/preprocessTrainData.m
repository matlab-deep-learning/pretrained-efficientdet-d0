function data = preprocessTrainData(data)
% Resize the images and scale the pixels to between 0 and 1. Also scale the
% corresponding bounding boxes.

% Copyright 2021 The MathWorks, Inc.

for ii = 1:size(data,1)
    I = data{ii,1};
    bboxes = data{ii,2};
    imgSize = size(I);
    
    % Convert an input image with single channel to 3 channels.
    if numel(imgSize) < 3 
        I = repmat(I,1,1,3);
    end
    
    % Preproces the input image and bounding boxes.
    [new_image, image_scale] = helper.preprocess(I);
    I = new_image;
    bboxes = bboxresize(bboxes,image_scale);
    
    data(ii, 1:2) = {I, bboxes};
end
end