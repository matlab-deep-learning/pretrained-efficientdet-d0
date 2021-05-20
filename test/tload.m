classdef(SharedTestFixtures = {DownloadEfficientDetD0Fixture}) tload < matlab.unittest.TestCase
    % Test for loading the downloaded models.
    
    % Copyright 2021 The MathWorks, Inc.
    
    % The shared test fixture DownloadEfficientDetD0Fixture calls
    % downloadPretrainedEfficientDetD0. Here we check that the properties of
    % downloaded models.
    
    properties        
        DataDir = fullfile(getRepoRoot(),'model');        
    end
    
    methods(Test)
        function verifyModelAndFields(test)
            % Test point to verify the fields of the downloaded models are
            % as expected.
                                    
            loadedModel = load(fullfile(test.DataDir,'efficientDetD0-coco.mat'));
            
            test.verifyClass(loadedModel.net,'dlnetwork');
            test.verifyEqual(numel(loadedModel.net.Layers),655);
            test.verifyEqual(size(loadedModel.net.Connections),[814 2]);
            test.verifyEqual(size(loadedModel.net.Learnables),[554 3]);
            test.verifyEqual(size(loadedModel.net.State),[126 3]);
            test.verifyEqual(loadedModel.net.InputNames,{'image_input'});            
        end        
    end
end