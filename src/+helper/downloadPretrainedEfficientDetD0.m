function model = downloadPretrainedEfficientDetD0()
% The downloadPretrainedEfficientDetD0 function loads a pretrained
% EfficientDet-D0 network.
%
% Copyright 2021 The MathWorks, Inc.

dataPath = 'model';
modelName = 'efficientDetD0-coco';
netFileFullPath = fullfile(dataPath, modelName);

% Add '.zip' extension to the data.
netFileFull = [netFileFullPath,'.zip'];

if ~exist(netFileFull,'file')
    fprintf(['Downloading pretrained', modelName ,'network.\n']);
    fprintf('This can take several minutes to download...\n');
    url = 'https://ssd.mathworks.com/supportfiles/vision/deeplearning/models/efficientDetD0/efficientDetD0-coco.zip';
    websave (netFileFullPath,url);
    unzip(netFileFullPath, dataPath);
    model = load([dataPath, '/efficientDetD0-coco.mat']);
else
    fprintf('Pretrained EfficientDet-D0 network already exists.\n\n');
    unzip(netFileFullPath, dataPath);
    model = load([dataPath, '/efficientDetD0-coco.mat']);
end
end