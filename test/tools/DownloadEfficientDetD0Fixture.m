classdef DownloadEfficientDetD0Fixture < matlab.unittest.fixtures.Fixture
    % DownloadEfficientDetD0Fixture   A fixture for calling downloadPretrainedEfficientDetD0 
    % if necessary. This is to ensure that this function is only called once
    % and only when tests need it. It also provides a teardown to return
    % the test environment to the expected state before testing.
    
    % Copyright 2021 The MathWorks, Inc
    
    properties(Constant)
        EfficientDetDataDir = fullfile(getRepoRoot(),'model')
    end
    
    properties
        EfficientDetExist (1,1) logical        
    end
    
    methods
        function setup(this)            
            this.EfficientDetExist = exist(fullfile(this.EfficientDetDataDir,'efficientDetD0-coco.mat'),'file')==2;
            
            % Call this in eval to capture and drop any standard output
            % that we don't want polluting the test logs.
            if ~this.EfficientDetExist
            	evalc('helper.downloadPretrainedEfficientDetD0();');
            end       
        end
        
        function teardown(this)
            if this.EfficientDetExist
            	delete(fullfile(this.EfficientDetDataDir,'efficientDetD0-coco.mat'));
            end            
        end
    end
end
