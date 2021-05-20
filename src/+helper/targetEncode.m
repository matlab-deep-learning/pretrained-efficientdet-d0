function target = targetEncode(G, P)
% Create regression targets.

% Copyright 2021 The MathWorks, Inc.

    % Center of proposal.
    px = P(:,1) + floor(P(:,3)/2);
    py = P(:,2) + floor(P(:,4)/2);

    % Center of groundtruth.
    gx = G(:,1) + floor(G(:,3)/2);
    gy = G(:,2) + floor(G(:,4)/2);
    
    % Encode targets.
    tx = (gx - px) ./(P(:,3));
    ty = (gy - py) ./(P(:,4));
    tw = log(G(:,3)./(P(:,3)));
    th = log(G(:,4)./(P(:,4)));
    
    % Observations in columns.
    target = [ty tx th tw]; 
end