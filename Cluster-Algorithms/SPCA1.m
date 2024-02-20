    function  [mappedX, mapping] = SPCA1(X, no_dims)
    disp('Computing eigenvectors of covariance matrix...');
    mapping = zeros(length(X(1,:)), no_dims);
    for e=1:no_dims
        fprintf('.');
        % Intialize eigenvector
        ev = repmat(1, [size(X, 2) 1]) * 0.01;

        % Compute mean feature vector
        s = repmat(0, [1 size(X, 2)]);
        for j=1:size(X, 1)
            s = s + X(j,:);
        end
        meanFV = s / size(X, 1);

        for j=1:size(X, 1)
            % Substract mean feature vector from features
            featureVector = X(j,:) - meanFV;
            featureVector = featureVector';

            % Deflate sample with already known eigenvectors
            for ei=1:(e - 1)
                if size(featureVector) == size(mapping(:,ei))
                    featureVector = featureVector - (mapping(:,ei)' * featureVector) .* mapping(:,ei);
                end
            end

            % Perform iterative SPCA step
            if size(featureVector) == size(ev)
                ev =iterative_spca(featureVector, ev);
            end
        end

        % Compute final eigenvector
        ev = ev / sqrt(ev' * ev);
        mapping(:,e) = ev;
    end


    % Apply mapping
    for i=1:size(X, 1)
        mappedX(i,:) = X(i,:) * mapping;
    end
end
function a_next = iterative_spca(x, a_current, output)
%ITERATIVE_SPCA Performs single SPCA iteration
%
%   a_next = iterative_spca(x, a_current, output)
%
% Performs single SPCA iteration. This function is used by the SPCA
% function.
%
%

% This file is part of the Matlab Toolbox for Dimensionality Reduction v0.2b.
% The toolbox can be obtained from http://www.cs.unimaas.nl/l.vandermaaten
% You are free to use, change, or redistribute this code in any way you
% want. However, it is appreciated if you maintain the name of the original
% author.
%
% (C) Laurens van der Maaten
% Maastricht University, 2007

    % Perform a single SPCA update
    s_x = size(x);
    n_features = s_x(1,1);
    y = a_current' * x;
    phi_2 = (1 / pdist([a_current'; zeros(1, n_features)])) * y * x;
    a_next = a_current + phi_2;
end