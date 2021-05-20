function [lgraph, networkOutputs] = configureEfficientDetD0(net, numClasses)
% Configure the pretrained network for transfer learning.

% Copyright 2021 The MathWorks, Inc.

% Extract the network output names.
networkOutputs = string(net.OutputNames');

% Extract the layergraph of the pretrained model. 
lgraph = layerGraph(net);

% Replace the final convolution layers to match the number of classes.
for k = 1:numel(networkOutputs)/2
    classLayerName = networkOutputs{2*(k-1)+1};
    boxLayerName = networkOutputs{2*k};
    lgraph = replaceLayer(lgraph, classLayerName, convolution2dLayer(1,(numClasses+1)*9,...
        "Name",classLayerName));
    lgraph = replaceLayer(lgraph, boxLayerName, convolution2dLayer(1,4*9,...
        "Name",boxLayerName));
end
end