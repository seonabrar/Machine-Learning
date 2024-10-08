% Load the data set from the file 'student_debt_data.csv'
data = load('student_debt_data.csv');
A = data';

% Construct the cost function
g = @(w) sum((A(:, 1) * w(2) + w(1) - A(:, 2)).^2, 1)';

% Set the step size
ALPHA = 1e-6;

% Set the upper limit of iterations
MAX_ITER = 1000;

% Set the starting point of iteration (Ensure it is at least 10 units away from the exact solution obtained in the previous problem)
w0 = [0, 1];

% Solve the weights using GD with AD
[gw, w, g_history, w_history] = gradientDescentAD(g, w0, ALPHA, MAX_ITER);

% Plot the result (not mandatory, but beneficial)
figure
subplot(211)
plot(g_history)
title('Cost history', 'r')
xlabel('Iteration number')
ylabel('Cost (g(w))')
set(gca, 'XScale', 'log')
set(gca, 'YScale', 'log')

subplot(212)
f = @(x, y) g([x y]);
fcontour(@(x, y) arrayfun(f, x, y), [-1000 1000 -100 100])
hold on
plot(w_history(:, 1), w_history(:, 2), 'r')
xlabel('w_1')
ylabel('w_2')
title('Cost contour and weight history')
