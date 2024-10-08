% Load the data from the file named 'ellipse_2class_data.csv'
A = load('ellipse_2class_data.csv');

% Separate features and labels
X = A(1:2, :)';  % Features
y = A(3, :)';    % Labels

% Call your training function
theta = fitNonlinearSoftmax(X, y);

function theta = fitNonlinearSoftmax( X, y )

    % << IMPLEMENT THE FUNCTION BODY! TYPICAL STEPS ARE GIVEN IN COMMENTS BELOW >>

    % Initialize variables
    P = size(X, 1);
    N = size(X, 2);
    theta0 = randn(1, N + 1); % Initial parameters
    alpha = 0.1; % Step size
    max_iter = 3000; % Maximum number of iterations

    % Solve the weights using GD with AD
    [~, theta, ~, ~] = gradientDescentAD(@cost, theta0, alpha, max_iter);

    % This function computes the Softmax cost function on nonlinear model
    % NOTE: As a nested function, it can use X and y directly and needs only the parameter vector theta
    function c = cost(theta)
        % Compute the Softmax cost
        y_bar = model(X, theta);
        c = 1/P * sum(log(1 + exp(-y .* y_bar)));
    end

end

% Local helper functions below

% This function transforms the features X non-linearly using the parameters v
function z = feature_transform( X, v )
    % No parameters needed for this transform so v not used!
    z = [ones(size(X,1), 1), X(:,1).^2, X(:,2).^2];
end

% This function applies the model specified by the parameters theta to the data X
function y = model(X, theta)
    % Transform features
    X_transformed = feature_transform(X, []);

    % Compute model output on transformed data with the given parameters
    y = X_transformed * theta';
end