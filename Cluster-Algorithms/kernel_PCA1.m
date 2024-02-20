    function [mappedX, mapping] = kernel_PCA1(X, no_dims, varargin)
    if ~exist('no_dims', 'var')
        no_dims = 2;
    end
    kernel = 'gauss';
    param1 = 1;
	param2 = 3;
    if nargin > 2
		kernel = varargin{1};
		if length(varargin) > 1 & strcmp(class(varargin{2}), 'double'), param1 = varargin{2}; end
		if length(varargin) > 2 & strcmp(class(varargin{3}), 'double'), param2 = varargin{3}; end
    end
    
    % Store the number of training and test points
    ell = size(X, 1);

    if size(X, 1) < 1000

        % Compute Gram matrix for training points
        disp('Computing kernel matrix...'); 
        K = gram(X, X, kernel, param1, param2);

        % Normalize kernel matrix K
        mapping.column_sums = sum(K) / ell;                       % column sums
        mapping.total_sum   = sum(mapping.column_sums) / ell;     % total sum
        J = ones(ell, 1) * mapping.column_sums;                   % column sums (in matrix)
        K = K - J - J' + mapping.total_sum * ones(ell, ell);
 
        % Compute first no_dims eigenvectors and store these in V, store corresponding eigenvalues in L
        disp('Eigenanalysis of kernel matrix...');
        K(isnan(K)) = 0;
        K(isinf(K)) = 0;
        [V, L] = eig(K);
    else
        % Compute column sums (for out-of-sample extension)
        mapping.column_sums = kernel_function([], X', 1, kernel, param1, param2, 'ColumnSums') / ell;
        mapping.total_sum   = sum(mapping.column_sums) / ell;
        
        % Perform eigenanalysis of kernel matrix without explicitly
        % computing it
        disp('Eigenanalysis of kernel matrix (using slower but memory-conservative implementation)...');
        options.disp = 0;
        options.isreal = 1;
        options.issym = 1;
        [V, L] = eigs(@(v)kernel_function(v, X', 1, kernel, param1, param2, 'Normal'), size(X, 1), no_dims, 'LM', options);
        disp(' ');
    end
    
    % Sort eigenvalues and eigenvectors in descending order
    [L, ind] = sort(diag(L), 'descend');
    L = L(1:no_dims);
	V = V(:,ind(1:no_dims));
    
    % Compute inverse of eigenvalues matrix L
	disp('Computing final embedding...');
    invL = diag(1 ./ L);
    
    % Compute square root of eigenvalues matrix L
    sqrtL = diag(sqrt(L));
    
    % Compute inverse of square root of eigenvalues matrix L
    invsqrtL = diag(1 ./ diag(sqrtL));
    
    % Compute the new embedded points for both K and Ktest-data
    mappedX = sqrtL * V';                     % = invsqrtL * V'* K
    
    % Set feature vectors in original format
    mappedX = mappedX';
    
    % Store information for out-of-sample extension
    mapping.X = X;
    mapping.V = V;
    mapping.invsqrtL = invsqrtL;
    mapping.kernel = kernel;
    mapping.param1 = param1;
    mapping.param2 = param2;
   function y = kernel_function(v, X, center, kernel, param1, param2, type)

    if ~exist('center', 'var')
        center = 0;
    end
    if ~exist('type', 'var')
        type = 'Normal';
    end
    if ~strcmp(type, 'ColumnSums'), fprintf('.'); end    
        
    % If no kernel function is specified
    if nargin == 2 || strcmp(kernel, 'none')
        kernel = 'linear';
    end
    
    % Construct result vector
    y = zeros(1, size(X, 1));
    n = size(X, 2);
    
    switch kernel
        
        case 'linear'
            % Retrieve information for centering of K
            if center || strcmp(type, 'ColumnSums')
                column_sum = zeros(1, n);
                for i=1:n
                    % Compute single row of the kernel matrix
                    K = X(:,i)' * X;
                    column_sum = column_sum + K;
                end
                % Compute centering constant over entire kernel
                total_sum = ((1 / n^2) * sum(column_sum));
            end
            
            if ~strcmp(type, 'ColumnSums')
                % Compute product K*v
                for i=1:n
                    % Compute single row of the kernel matrix
                    K = X(:,i)' * X;

                    % Center row of the kernel matrix
                    if center
                        K = K - ((1 / n) .* column_sum) - ((1 / n) .* column_sum(i)) + total_sum;
                    end

                    % Compute sum of products
                    y(i) = K * v;
                end
            else
                % Return column sums
                y = column_sum;
            end
            
        case 'poly'
            
            % Initialize some variables
            if ~exist('param1', 'var'), param1 = 1; param2 = 3; end            
                        
            % Retrieve information for centering of K
            if center || strcmp(type, 'ColumnSums')
                column_sum = zeros(1, n);
                for i=1:n
                    % Compute row sums of the kernel matrix
                    K = X(:,i)' * X;
                    K = (K + param1) .^ param2;
                    column_sum = column_sum + K;
                end
                % Compute centering constant over entire kernel
                total_sum = ((1 / n^2) * sum(column_sum));
            end       

            if ~strcmp(type, 'ColumnSums')
                % Compute product K*v
                for i=1:n
                    % Compute row of the kernel matrix
                    K = X(:,i)' * X;
                    K = (K + param1) .^ param2;

                    % Center row of the kernel matrix
                    if center
                        K = K - ((1 / n) .* column_sum) - ((1 / n) .* column_sum(i)) + total_sum;
                    end

                    % Compute sum of products
                    y(i) = K * v;
                end
            else
                % Return column sums
                y = column_sum;
            end
            
        case 'gauss'
            
            % Initialize some variables
            if ~exist('param1', 'var'), param1 = 1; end
            
            % Retrieve information for centering of K
            if center || strcmp(type, 'ColumnSums')
                column_sum = zeros(1, n);
                for i=1:n
                    % Compute row sums of the kernel matrix
                    K = L2_distance(X(:,i), X);
                    K = exp(-(K.^2 / (2 * param1.^2)));
                    column_sum = column_sum + K;
                end
                % Compute centering constant over entire kernel
                total_sum = ((1 / n^2) * sum(column_sum));
            end          

            if ~strcmp(type, 'ColumnSums')
                % Compute product K*v
                for i=1:n
                    % Compute single row of the kernel matrix
                    K = L2_distance(X(:,i), X);
                    K = exp(-(K.^2 / (2 * param1.^2)));

                    % Center row of the kernel matrix
                    if center
                        K = K - ((1 / n) .* column_sum) - ((1 / n) .* column_sum(i)) + total_sum;
                    end

                    % Compute sum of products
                    y(i) = K * v;                    
                end
            else
                % Return column sums
                y = column_sum;
            end
            
        otherwise
            error('Unknown kernel function.');
    end 
   end
    end

    function G = gram(X1, X2, kernel, param1, param2)
%GRAM Computes the Gram-matrix of data points X using a kernel function
%
%   G = gram(X1, X2, kernel, param1, param2)
%
% Computes the Gram-matrix of data points X1 and X2 using the specified kernel
% function. If no kernel is specified, no kernel function is applied. The
% function GRAM is than equal to X1*X2'. The use of the function is different
% depending on the specified kernel function (because different kernel
% functions require different parameters. The possibilities are listed
% below.
% Linear kernel: G = gram(X1, X2, 'linear')
%           which is parameterless
% Gaussian kernel: G = gram(X1, X2, 'gauss', s)
%           where s is the variance of the used Gaussian function (default = 1).
% Polynomial kernel: G = gram(X1, X2, 'poly', R, d)
%           where R is the addition value and d the power number (default = 0 and 3)
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

    % Check inputs
    if size(X1, 2) ~= size(X2, 2)
        error('Dimensionality of both datasets should be equal');
    end

    % If no kernel function is specified
    if nargin == 2 || strcmp(kernel, 'none')
        kernel = 'linear';
    end
    
    switch kernel
        
        % Linear kernel
        case 'linear'
            G = X1 * X2';
        
        % Gaussian kernel
        case 'gauss'
            if ~exist('param1', 'var'), param1 = 1; end
            G = L2_distance(X1', X2');
            G = exp(-(G.^2 / (2 * param1.^2)));
                        
        % Polynomial kernel
        case 'poly'
            if ~exist('param1', 'var'), param1 = 1; param2 = 3; end
            G = ((X1 * X2') + param1) .^ param2;
            
        otherwise
            error('Unknown kernel function.');
    end
    end