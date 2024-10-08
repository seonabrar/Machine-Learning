% Create (scalar) variable 'a' and store the value 2.3 into it. For this first problem, the correct solution is shown below
a = 2.3;

% Create a 1x4 row vector 'v' with the elements 6, 2, 9, and 11, in this order
v = [6, 2, 9, 11];

% Create a 50x1 column vector 'w' with the elements from 1 to 50 in the increasing order
w = (1:50)';

% Create a 3x4 matrix 'X' with elements i*j at the row index i=1,2,3; and column index j=1,...,4
X = zeros(3, 4);

for i = 1:3
    for j = 1:4
        X(i, j) = i * j;
    end
end

% Using matrix indexing, pick the element of X at the row index 2, and column index 3 into a new variable named x_2_3
x_2_3 = X(2, 3);

% Using matrix indexing, take the entire first row of X, and store it as the vector called x1
x1 = X(1, :);

% Using matrix indexing, take the entire second column of X, and store it as the vector called x2
x2 = X(:, 2);

% Using matrix indexing, take the submatrix consisting of the elements in the first and seconds rows and columns, and store it as the matrix Xs
Xs = X(1:2, 1:2);

% Calculate the vector 'u' as the product of the matrix X and the transpose of the vector v.
u = X * v';

% Add a constant value 2 to all the elements of w and store the result as the vector called 'w2'
w2 = w + 2;

% Load the file 'problem2.mat' using the load function without assigning its output to any variable. The workspace will then contain the vector 'v' of integers needed for this problem
load('problem2.mat');
% Write a for loop that computes the sum of the elements in v, and stores it to the variable 's'
s = 0;

for i = 1:length(v)
    s = s + v(i);
end
% Calculate the same sum using the 'sum' function instead of a loop, and store the result into 'ss'
ss = sum(v);

% Calculate the sum of squares of the elements in v using the 'sum' function and the element-wise power operator (.^) instead of a loop. Store the result into 'ss2'
ss2 = sum(v.^2);

% Make another for loop that goes through the elements of v, and creates a same size vector called 'w' containing -1 for odd, and +1 even valued element indexes of v.
% Use the if-else construct to choose which one (-1 or +1) to pick for each element
w = zeros(size(v)); % Initialize the vector 'w' with zeros

for i = 1:length(v)
    if mod(i, 2) == 1
        % Odd index, set the corresponding element of 'w' to -1
        w(i) = -1;
    else
        % Even index, set the corresponding element of 'w' to +1
        w(i) = 1;
    end
end

% Similarly to previous but instead of a for loop, create the vector 'w2' by artihmetically manipulating the output of the modulo function ('mod') on the whole vector v

w2 = (-1).^(mod(1:length(v), 2));

% Find the maximum value of the elements of v and store it into variable 'v_max'
v_max = max(v);

% Find the index (not the value) of the minimum valued element of v into variable 'v_min_loc'. In case of multiple minima, store only the location of the first one.
[~, v_min_loc] = min(v);

