% If necessary, define local helper functions below

% Load the data set from the file 'linear_2output_regression.csv'
A = load('linear_2output_regression.csv');

% Input variables as a matrix with samples on rows
X = A(1:2, :)';

% Output variables as a matrix with samples on rows
Y = A(3:end, :)';

% Call your fitting function
W = fitMultipleOutputRegression(X, Y);

% Plot the result (not mandatory, but beneficial)
xx = 0:0.1:1;
[XX, YY] = meshgrid(xx);

C = size(Y, 2);

figure
for i = 1:C
    subplot(C, 1, i)

    scatter3(X(:, 1), X(:, 2), Y(:, i), 'filled', 'k')
    view(25, 25)
    xlabel('x_1')
    ylabel('x_2')
    zlabel(sprintf('y_%d', i))
    title(sprintf('Plot of output %d samples and the fitted plane', i))

    hold on

    ZZ = arrayfun(@(x, y) [1, x, y] * W(:, i), XX, YY);
    surf(XX, YY, ZZ)
end


function W = fitMultipleOutputRegression(X, Y)
    % Initialize variables and set up the problem(s)
    P = size(X, 1); % Number of samples
    C = size(Y, 2); % Number of outputs

    X0 = [ones(P, 1), X]; % Augmented input matrix
    alpha = 0.01; % Set an appropriate learning rate
    max_iter = 2000; % Set an appropriate maximum number of iterations

    % Initialize the output weight matrix
    W = zeros(size(X0, 2), C);

    % Define the cost function for single output variable regression
    cost = @(w, X, y) (1/P) * sum(abs(X * w' - y));

    % Solve subproblems separately in a loop
    for i = 1:C
        % Use gradientDescentAD for the subproblem i
        w0 = zeros(1, size(X0, 2)); % Initialize weights
        g = @(w) cost(w, X0, Y(:, i)); % Define cost function for subproblem i

        [gw, w, ~, w_history] = gradientDescentAD(g, w0, alpha, max_iter);

        % Store the result in the corresponding column of the output weight matrix
        W(:, i) = w_history(end, :);
    end
end
