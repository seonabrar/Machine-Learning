% Load the data from 'regression_outliers.csv' using the load function
data = load('regression_outliers.csv');
A = data(:, 1:10);

% Load the weights from 'problem1.mat' using the load function
weights = load('problem1.mat');
w_LS = weights.w_LS;

% Construct the Least Squares cost function
X = [ones(size(A, 2), 1), A(1, :)'];
y = A(2, :)';
cost_LS = @(w) sum((X * w - y).^2);
% Construct the Least Absolute Deviations cost function
cost_LAD = @(w) sum(abs(X * w - y));

% Compute the LS cost on weights w_LS
cost_LS_wLS = cost_LS(w_LS);
cost_LS_wLAD = cost_LS(w_LAD);

% Compute the LAD cost on weights w_LS and w_LAD
cost_LAD_wLS = cost_LAD(w_LS);
cost_LAD_wLAD = cost_LAD(w_LAD);

% Create 100 evenly spaced grid of points between -2 and 2 (inclusive) for model evaluation and plotting. Create a column vector
x = linspace(-2, 2, 100)';

% Evaluate the LS model at x, i.e., use w_LS to calculate output at the points in x. Create a column vector of results
y_LS = [ones(size(x)), x] * w_LS;

% Evaluate the LAD model at x, i.e., use w_LAD to calculate output at the points in x. Create a column vector of results
y_LAD = [ones(size(x)), x] * w_LAD;

% Plot the result
figure
scatter(A(1,:), A(2,:))
hold on
plot(x, y_LS, 'LineWidth', 2)
plot(x, y_LAD, '--', 'LineWidth', 2)
legend('data', 'Least Squares', 'Least Absolute Deviations', 'Location', 'NorthWest')
axis([-2 2 -5 12])
xlabel('x')
ylabel('y')
title('Comparison of LS and LAD Models with Outlier')
