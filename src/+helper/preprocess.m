function [new_image, image_scale] = preprocess(image)
% This function preprocesses the input image.

% Copyright 2021 The MathWorks, Inc.

% Normalize the input image.
I = im2single(image);
meanIm = [0.485 0.456 0.406];
stdIm = [0.229 0.224 0.225];
I = (I - reshape(meanIm,[1 1 3]))./reshape(stdIm,[1 1 3]);

% Compute scale factors.
[image_h,image_w,~]= size(I);
image_scale_y = 512/image_h;
image_scale_x = 512/image_w;
image_scale = min(image_scale_x,image_scale_y);
scaled_height = single(image_h * image_scale);
scaled_width = single(image_w * image_scale);

% Resize image to size [512 512].
scaled_image = imresize(I,[scaled_height,scaled_width],'method','bilinear');
new_image = padarray(scaled_image,abs(size(scaled_image,1:2)-[512 512]),'post');
end