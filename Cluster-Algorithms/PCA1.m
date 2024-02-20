function [mappedX, mapping] = PCA1(X, no_dims)
    if ~exist('no_dims', 'var')
        no_dims = 2;
    end
	
	% Make sure data is zero mean
    mapping.mean = mean(X, 1);
	X = X - repmat(mapping.mean, [size(X, 1) 1]);
	% Compute covariance matrix
	C = cov(X);
	% Perform eigendecomposition of C
	C(isnan(C)) = 0;
	C(isinf(C)) = 0;
    [M, lambda] = eig(C); 
    % Sort eigenvectors in descending order
    [lambda, ind] = sort(diag(lambda), 'descend');
	M = M(:,ind(1:no_dims));	
	% Apply mapping on the data
	mappedX = X * M;%降维后的X
    % Store information for out-of-sample extension
    mapping.M = M;%映射的基
	mapping.lambda = lambda;
    end