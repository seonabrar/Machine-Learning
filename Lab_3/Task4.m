% Load the data set
A = load('noisy_sin_sample.csv');

% Step size
ALPHA = 1e-1;

% Upper limit of iterations
MAX_ITER = 2000;

% Initial point
theta0 = [2, -4, 1, 2];

[theta, cost_history, theta_history] = fitSingleOutputRegression(A(:, 1), A(:, 2), theta0, ALPHA, MAX_ITER);

% Plot the result (not mandatory, but beneficial)
figure
subplot(211)
plot(cost_history)
title('Cost history', 'r')
xlabel('Iteration number')
ylabel('Cost (g(w))')
set(gca, 'XScale', 'log')
set(gca, 'YScale', 'log')

subplot(212)
plot(theta_history)
title('Theta history', 'r')
xlabel('Iteration number')
ylabel('Parameter value')

% Add a plot to visualize the fit of the model on the noisy sinusoidal data
figure
scatter(A(:, 1), A(:, 2), 'b.'); hold on;
x_vals = linspace(min(A(:, 1)), max(A(:, 1)), 100)';
y_vals = model(x_vals, theta);
plot(x_vals, y_vals, 'r-', 'LineWidth', 2);
title('Noisy Sinusoidal Data and Fitted Model');
xlabel('X');
ylabel('Y');
legend('Data', 'Fitted Model');
grid on;

% The main function to do the non-linear fitting
function [theta, cost_history, theta_history] = fitSingleOutputRegression(X, y, theta0, alpha, max_iter)

    % Initialize variables
    theta = theta0;
    cost_history = zeros(max_iter, 1);
    theta_history = zeros(max_iter, length(theta0));

    % Solve the weights using GD with AD
    [gw, w, g_history, w_history] = gradientDescentAD(@cost, theta0, alpha, max_iter);

    % Return the best solution, and the histories
    theta = w;
    cost_history = g_history;
    theta_history = w_history;

    % This function computes the least squares cost function
    % NOTE: As a nested function, it can use X and y directly and needs only the parameter vector theta
    function c = cost(theta)
        m = length(X);
        predictions = model(X, theta);
        residuals = predictions - y;
        c = 0.5 * sum(residuals.^2) / m;
    end
end

% Local helper functions below

% This function transforms the features x non-linearly using the parameters v
function transformed_features = feature_transform(x, v)
    transformed_features = [ones(size(x)), sin(v(1) + v(2) * x)]; % << COMPUTE THE TRANSFORM >>
end

% This function applies the model specified by the parameters theta to the data x
function y = model(x, theta)
    transformed_features = feature_transform(x, theta(3:end));
    y = transformed_features * theta(1:2)'; % << COMPUTE MODEL OUTPUT ON TRANSFORMED DATA WITH THE GIVEN PARAMETERS >>
end


