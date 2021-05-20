function updatePlots(lossPlotter, learningRatePlotter, iteration, currentLR, totalLoss)
% Update loss and learning rate plots.

% Copyright 2021 The MathWorks, Inc.

loss = double(extractdata(gather(totalLoss)));
addpoints(lossPlotter, iteration, loss);
addpoints(learningRatePlotter, iteration, currentLR);
drawnow
end