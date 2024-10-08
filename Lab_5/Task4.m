% Load the data from the file named '5class_data.csv'
A = load('5class_data.csv');

% Form the feature matrix starting from the first row upto second last
X = A(1:end-1,:)';

% Form the class column vector from the last row in the data
y = A(end,:)';

% Number of classes
C = 5;

% Call the training function
W = trainMultiClassSoftmax(X,y,C);

% This function is the one implemented in the first problem
c = classifyMultiClass( W, X );

accuracy = 100 * sum( c==y ) / length(c)

figure
scatter( X(:,1), X(:,2), 25, y, 'filled' )
hold on
scatter( X(:,1), X(:,2), 60, c )
axis([0 1 0 1])
xlabel('x_1')
ylabel('x_2')
title( sprintf('Classification accuracy %.2f %%', accuracy ) )

function W = trainMultiClassSoftmax(X, y, C)
    % Initialize variables
    P = size(X, 1); 
    N = size(X, 2); 
    ALPHA = 0.1; 
    MAX_ITER = 3000; 
    X0 = [ones(P, 1), X]; % Augmented input matrix
    w0 = randn(1, C*(N+1)); % Initial weight vector

    % Perform gradient descent on the cost_softmax function
    [cost_min, w_min, cost_history, w_history] = gradientDescentAD(@cost_softmax, w0, ALPHA, MAX_ITER);
    
    % Return the best weight vector but in matrix form
    best_weight = w_min;
    W = reshape(best_weight, C, N+1);
    
    % Nested cost function
    function c = cost_softmax(w)
        % For computations, transform w into matrix form
        new_W = reshape(w, C, N+1);
        % Evaluate the Multi-Class Softmax cost
        exp_sum = sum(exp(X0 * new_W'), 2);
        log_sum = log(exp_sum);
        correct_weights = new_W(y + 1, :);
        c = -sum(log_sum - sum(X0 .* correct_weights, 2)) / P;
    end
end
