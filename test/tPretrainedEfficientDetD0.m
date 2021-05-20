classdef(SharedTestFixtures = {DownloadEfficientDetD0Fixture}) tPretrainedEfficientDetD0 < matlab.unittest.TestCase
    % Test for tPretrainedEfficientDetD0
    
    % Copyright 2021 The MathWorks, Inc.
    
    % The shared test fixture downloads the model. Here we check the
    % inference on the pretrained model.
    properties        
        RepoRoot = getRepoRoot;
        ModelName = 'efficientDetD0-coco.mat';
    end
    
    methods(Test)
        function exerciseDetection(test)            
            model = load(fullfile(test.RepoRoot,'model',test.ModelName));
            image = imread('visionteam.jpg');
            classNames = helper.getCOCOClasess;
            
            expectedBboxes = single([385.30396,48.153008,117.53223,323.05319;...
                        152.20755,35.156727,110.23917712,372.80170;...
                        34.285458,52.577759,134.18510,360.46387;...
                        265.99649,36.726273,120.34817,349.86322;...
                        509.82104,58.244373,136.06734,341.60858;...
                        657.99353,55.037750,125.69208,357.13699]);
            expectedScores = single([0.76683837;0.74082363;0.82276124;0.85276574;0.86604726;0.89708591]);
            expectedLabels = categorical({'person';'person';'person';'person';'person';'person'});

            executionEnvironment = 'auto';
            [bboxes,scores,labels] = detectEfficientDetD0(model.net, image, classNames, executionEnvironment);
            
            % verifying bboxes from detectEfficientDetD0.
            test.verifyEqual(bboxes,expectedBboxes);
            % verifying scores from detectEfficientDetD0.
            test.verifyEqual(scores,expectedScores);
            % verifying labels from detectEfficientDetD0.
            test.verifyEqual(labels,expectedLabels);            
        end       
    end
end