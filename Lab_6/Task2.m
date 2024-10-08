% Load the data
A = load('credit_dataset.csv');

% Features are on the rows; up to the second last
X = A(1:end-1,:)';

% Normalize the data for learning
X = normalize(X);

% Class label is on the last row
y = A(end,:)';

% The set of regularization parameters to test
LAMBDAS = [0 0.01 0.02 0.04]';

% Perform l1-regularized Softmax training
[W, cost_history] = regularizedFeatureSelection(X, y, LAMBDAS);

% Plot the result (not mandatory, but beneficial)
N = size(X,2);          % Number of features
L = length(LAMBDAS);    % Number of lambdas
h = [];
figure
for i = 1:L
    h(i) = subplot(L,1,i);
    bar( 0:N, W(i,:) )
    xticks( 0:N+1 )
    title( sprintf('Weight values at lambda %.2f', LAMBDAS(i)), sprintf('Cost = %.2f', cost_history(i) ) )
end
linkaxes(h)


function [W, cost_history] = regularizedFeatureSelection(X, y, lambdas)
    % Initialize variables
    P = size(X, 1); % Number of samples
    N = size(X, 2); % Number of features
    w0 = randn(1, N + 1); % Initial weights
    W = zeros(length(lambdas), N + 1); % Matrix to store weights for each lambda
    cost_history = zeros(length(lambdas), 1); % Vector to store cost history
    
    % Number of regularization parameters lambda to test
    L = length(lambdas);
    
    % Go through all the lambdas
    for i = 1:L
        lambda = lambdas(i);
        
        % Call the local function to train perceptron with l1 regularization
        [w, cost] = trainPerceptronL1(X, y, lambda, w0);
        
        % Return trained weights in the matrix W
        W(i, :) = w;

        % Return the training cost in the cost_history
        cost_history(i) = cost;
    end
end

function [w, cost] = trainPerceptronL1(X, y, lambda, w0)
    % Initialize problem
    P = size(X, 1); % Number of samples
    N = size(X, 2); % Number of features
    X0 = [ones(P, 1), X]; % Augmented input data
    ALPHA = 0.1; % Learning rate
    MAX_ITER = 1000; % Maximum iterations
    
    % Perform gradient descent using AD
    [~, w_history, ~, cost_history] = gradientDescentAD(@cost_softmax, w0, ALPHA, MAX_ITER);
    
    % Choose the weights corresponding to the minimum cost
    [~, min_index] = min(cost_history);
    w = w_history(min_index, :);
    cost = cost_history(min_index);
    
    % L1-regularized Softmax cost function
    function c = cost_softmax(w)
        % Compute Softmax cost with l1 regularization
        h = X0 * w'; % Hypothesis
        exp_h = exp(-h); % Exponential term
        cost = log(1 + exp(exp_h)); % Softmax cost
        regularization_term = lambda * sum(abs(w(2:end))); % L1 regularization term (excluding bias)
        c = 1/P * sum(cost) + regularization_term; % Total cost
    end
end

