% Load the data from the file named '3class_data.csv'
A = load('3class_data.csv');

% Form the feature matrix starting from the first row up to the second last
X = A(1:end-1, :)';

% Form the class column vector from the last row in the data
y = A(end, :)';

% Number of classes
C = 3;

% Call the training function
W = trainMultiClassPerceptron(X, y, C);


function W = trainMultiClassPerceptron(X, y, C)

    % Initialize variables
    P = size(X, 1); 
    N = size(X, 2); 
    
    ALPHA = 0.1; 
    MAX_ITER = 3000; 
    
    X0 = [ones(P, 1), X]; % Augmented input matrix
    w0 = randn(1, C*(N+1)); % Initial weight vector

    
    % Perform gradient descent on the cost_perceptron function
    [cost_min, w_min, cost_history, w_history] = gradientDescentAD( @cost_perceptron, w0, ALPHA, MAX_ITER );
    
    % Return the best weight vector but in matrix form
    best_weight = w_min;
    W = reshape(best_weight, C, N+1)';
    
    % Nested cost function
    function c = cost_perceptron(w)

        % For computations, transform w into matrix form
        new_W = reshape(w, C, N+1);
        % Evaluate the Multi-Class Perceptron cost
        max_term = max(X0 * new_W', [], 2);
        correct_weights = new_W(y + 1, :);
        summation = sum(max_term - sum(X0 .* correct_weights, 2));
        c = 1/P * summation;
    end

end
