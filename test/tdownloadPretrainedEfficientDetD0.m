classdef(SharedTestFixtures = {DownloadEfficientDetD0Fixture}) tdownloadPretrainedEfficientDetD0 < matlab.unittest.TestCase
    % Test for downloadPretrainedEfficientDetD0
    
    % Copyright 2021 The MathWorks, Inc.
    
    % The shared test fixture DownloadEfficientDetD0Fixture calls
    % downloadPretrainedEfficientDetD0. Here we check that the downloaded files
    % exists in the appropriate location.
    
    properties        
        DataDir = fullfile(getRepoRoot(),'model');
    end
    
    methods(Test)
        function verifyDownloadedFilesExist(test)
            dataFileName = 'efficientDetD0-coco.mat';
            test.verifyTrue(isequal(exist(fullfile(test.DataDir,dataFileName),'file'),2));
        end
    end
end
