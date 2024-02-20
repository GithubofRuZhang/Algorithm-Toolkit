function [mappedX, mapping] = isomap1(X, no_dims, k)

    if ~exist('no_dims')
        no_dims = 2;
    end
    if ~exist('k')
        k = 12;
    end

    % Construct neighborhood graph
    disp('Constructing neighborhood graph...'); 
    D = find_nn(X, k);

    % Select largest connected component
    blocks = components(D)';
    count = zeros(1, max(blocks));
    for i=1:max(blocks)
        count(i) = length(find(blocks == i));
    end
    [count, block_no] = max(count);
    conn_comp = find(blocks == block_no);
    D = D(conn_comp,:);
    D = D(:,conn_comp);
    n = size(D, 1);

    % Compute shortest paths
    disp('Computing shortest paths...');
    D = dijkstra(D, [1:n]);
    
    % Performing MDS using eigenvector implementation
    disp('Constructing low-dimensional embedding...');
    M = -.5 * (D .^ 2 - sum(D .^ 2)' * ones(1, n) / n - ones(n, 1) * sum(D .^ 2) / n + sum(sum(D .^ 2)) / (n ^ 2));
	M(isnan(M)) = 0;
	M(isinf(M)) = 0;
    [vec, val] = eig(M);
	if size(vec, 2) < no_dims
		no_dims = size(vec, 2);
		warning(['Target dimensionality reduced to ' num2str(no_dims) '...']);
	end
	
    % Computing final embedding
    h = real(diag(val)); 
    [foo, sorth] = sort(h, 'descend');  
    val = real(diag(val(sorth, sorth))); 
    vec = vec(:,sorth);
    mappedX = real(vec(:,1:no_dims) .* (ones(n, 1) * sqrt(val(1:no_dims))')); 
    
    % Store data for out-of-sample extension
    mapping.conn_comp = conn_comp;
    mapping.k = k;
    mapping.X = X(conn_comp,:);
    mapping.D = D;
    mapping.vec = vec;
    mapping.val = val;
end
function [D, ni] = find_nn(X, k, type)
%FIND_NN Finds k nearest neigbors for all datapoints in the dataset
%
%	[D, ni] = find_nn(X, k, type)
%
% Finds the k nearest neighbors for all datapoints in the dataset X.
% In X, rows correspond to the observations and columns to the
% dimensions. The value of k is the number of neighbors that is
% stored. The function returns a sparse distance matrix D, in which
% only the distances to the k nearest neighbors are stored. For
% equal datapoints, the distance is set to a tolerance value.
% The method is relatively slow, but has a memory requirement of O(nk).
% Type can be 'eucl' (default) or 'sqeucl'.
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

	if ~exist('k', 'var')
		k = 12;
    end
    if ~exist('type', 'var')
		type = 'eucl';
    end

    % Memory conservative implementation
    if size(X, 1) > 2000
        X = X';
        n = size(X, 2);
        D = sparse(n, n);
        XX = sum(X .* X);
        if nargout > 1, ni = uint16(zeros(n, k)); end
        for i=1:n
            xx = sum(X(:,i) .* X(:,i));  
            xX = X(:,i)' * X;
            if strcmp(type, 'eucl')
                d = real(sqrt(repmat(xx', [1 size(XX, 2)]) + repmat(XX, [size(xx, 2) 1]) - 2 * xX));
            else
                d = abs(repmat(xx', [1 size(XX, 2)]) + repmat(XX, [size(xx, 2) 1]) - 2 * xX);
            end
            [d, ind] = sort(d);
            d = d(1:k); ind = ind(1:k);
            d(d == 0) = 1e-7;
            D(i, ind) = d;
            if nargout > 1, ni(i,:) = ind; end
        end

    % Faster implementation
    else
        n = size(X, 1);
        if nargout > 1, ni = uint16(zeros(n, k)); end
        if strcmp(type, 'eucl')
            D = squareform(pdist(X, 'euclidean'));
        else
            D = squareform(pdist(X, 'seuclidean'));
        end
        [foo, ind] = sort(D, 2);
        for i=1:n
            D(i, ind(i, k+1:end)) = 0;
            D(i,i) = 1e-7;
            if nargout > 1, ni(i,:) = ind(i, 1:k); end
        end
        D = sparse(double(D));
    end
end
function D = dijkstra( G , S )

N = size( G , 1 );
D = dijk( G , S , 1:N );
end
