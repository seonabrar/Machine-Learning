% Load the data set into matrix A from 'student_debt_data.csv' using the load function
data = load('student_debt_data.csv');
A = data';

% Construct the design matrix X with augmented ones
X = [ones(size(A, 1), 1), A(:, 1)];

% Construct the expected outcome vector y
y = A(:, 2);

% Solve the weights using Pseudoinverse
w = pinv(X) * y;

% Use the model to extrapolate year 2030 debt
year2030 = 2030;
debt2030 = [1, year2030] * w;

% Display the solution
disp('Weights (w):');
disp(w);
fprintf('Estimated student debt in 2030: $%.2f trillion\n', debt2030);

% Visualization
scatter(A(:, 1), A(:, 2), 'o', 'DisplayName', 'Data');
hold on
plot(A(:, 1), X * w, '-r', 'LineWidth', 2, 'DisplayName', 'Linear Regression');
plot(year2030, debt2030, 'xg', 'MarkerSize', 10, 'LineWidth', 2, 'DisplayName', 'Extrapolation (2030)');
legend('Location', 'NorthWest');
xlabel('Year');
ylabel('Student Debt (trillions of dollars)');
title('Linear Regression and Extrapolation of Student Debt');
grid on
