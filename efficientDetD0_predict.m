function out = efficientDetD0_predict(matFile, image)
%#codegen
% Copyright 2021 The MathWorks, Inc.

% Convert input to dlarray.
dlInput = dlarray(image, 'SSCB');

persistent efficientDetD0obj;

if isempty(efficientDetD0obj)
    efficientDetD0obj = coder.loadDeepLearningNetwork(matFile);
end

% Pass input.
out = cell(10,1);
[out{:}] = efficientDetD0obj.predict(dlInput);
end