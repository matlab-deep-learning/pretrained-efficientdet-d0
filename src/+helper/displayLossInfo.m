function displayLossInfo(epoch, iteration, currentLR, lossInfo)
% Display loss information for each iteration.

% Copyright 2021 The MathWorks, Inc.

disp("Epoch : " + epoch + " | Iteration : " + iteration  + " | Learning Rate : " + currentLR + ...
    " | Total Loss : " + double(gather(extractdata(lossInfo.totalLoss))) + ...
    " | Box Loss : " + double(gather(extractdata(lossInfo.loc_Loss))) + ...
    " | Class Loss : " + double(gather(extractdata(lossInfo.conf_loss))));
end