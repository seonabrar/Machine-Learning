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
    
    X0 = [ones(P, 1), X];
    w0 = randn(1, N + 1); 
    alpha = 0.01;
    max_iter = 3000;

    % Solve the weights using GD with AD, call [...] = gradientDescentAD(...)
    [~, w, ~, ~] = gradientDescentAD(@cost, w0, alpha, max_iter);

    % Return the best solution, and the histories
    theta = w; % Corrected: Added semicolon
    % This function computes the Softmax cost function on nonlinear model
    % NOTE: As a nested function, it can use X and y directly and needs only the parameter vector theta
    function c = cost(theta)
        % Compute the Softmax cost
        y_bar = model(X0, theta);
        u = -y .* y_bar;
        c = 1/P * sum(log(1 + exp(u)));
    end

end

% Local helper functions below

% This function transforms the features X non-linearly using the parameters v
function z = feature_transform( X, v )
    % No parameters needed for this transform so v not used!
    z = [ones(size(X, 1), 1), X(:,1).^2, X(:,2).^2];
end

% This function applies the model specified by the parameters theta to the data X
function y = model(X, theta)
    y = feature_transform(X, []) * theta';
end
