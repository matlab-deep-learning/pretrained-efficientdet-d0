function anchorBoxes = generateAnchorBox()
% Generate multi scale anchor boxes for different feature levels.

% Copyright 2021 The MathWorks, Inc.

minLevel = 3;
maxLevel = 7;
numScales = 3;
anchorScale = 4;
aspectRatios = [1 2 0.5];
inputSize = [512 512];
featSizes = inputSize/128 .* 2.^((maxLevel-minLevel):-1:0)';
strides = inputSize(1:2)./featSizes;
numOfBoxesEachLevel = numScales*numel(aspectRatios);
numOfGrid = sum(prod(featSizes,2),2);
numOfBoxes = sum(numOfGrid)*numOfBoxesEachLevel;
octaveScales = (0:(numScales-1))/numScales;
[aspectRatiosGrid,octaveScalesGrid] = ndgrid(aspectRatios,2.^octaveScales);
aspectRatiosGrid = aspectRatiosGrid(:);
octaveScalesGrid = octaveScalesGrid(:);
aspectX = sqrt(aspectRatiosGrid);
aspectY = 1./aspectX;
anchorUnitSizeX = anchorScale .* octaveScalesGrid(:) .* aspectX;
anchorUnitSizeY = anchorScale .* octaveScalesGrid(:) .* aspectY;
boxScalesAct = zeros(numOfBoxes,2);
boxOffsetsAct = zeros(numOfBoxes,2);
indPre = 1;
for k = 1:(maxLevel - minLevel + 1)
    anchorSizeX = anchorUnitSizeX * strides(k,2);
    anchorSizeY = anchorUnitSizeY * strides(k,1);
    x = (strides(k,2)/2:strides(k,2):inputSize(2))';
    y = (strides(k,1)/2:strides(k,2):inputSize(1))';
    [xv,yv] = ndgrid(x,y);
    indPost = indPre + numel(xv) * numOfBoxesEachLevel -1;
    boxScalesAct(indPre:indPost,:) = repmat([anchorSizeY,anchorSizeX],[numel(xv),1]);
    boxOffsetsAct(indPre:indPost,:) = repelem([yv(:) xv(:)],numOfBoxesEachLevel,1);
    indPre = indPost + 1;
end
anchorBoxes = [boxOffsetsAct boxScalesAct];
end