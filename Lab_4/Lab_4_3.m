% Load X and y from 'spambase_data.mat'
load('spambase_data.mat');

% Call your training function
w = trainPerceptronNormalized(X, y);

function w = trainPerceptronNormalized(X,y)

    % Initialize variables and set up the problem: initial parameters, cost function etc.
    P = size(X, 1);
    N = size(X, 2);
    
    X0 = normalize(X);

    X0 = [ones(P, 1), X0]; % Add bias term after normalization
    w0 = randn(1, N + 1); 
    alpha = 0.1;
    max_iter = 3000;

    % Solve the problem using gradientDescentAD
    [cost_min, w_min, cost_history, w_history] = gradientDescentAD(@cost_perceptron, w0, alpha, max_iter);

    
    % If necessary, define nested helper functions below
    w = w_min;


    % Define the Perceptron cost function
    function c = cost_perceptron(w)
        % Compute the perceptron cost
        y_bar = X0 * w';
        u = y.* y_bar;
        v = exp(-u);
        s = log(1+v);
        c = 1/P * sum(s);
    end
     % Initialize accuracy history
    accuracy_history = zeros(max_iter, 1);

    % Loop through each iteration to calculate accuracy
    for i = 1:max_iter
        % Predict labels using the trained perceptron weights w
        y_pred = sign(X0 * w_history(i,:)');

        % Calculate the number of correct predictions
        correct_predictions = sum(y_pred == y');

        % Calculate the accuracy
        accuracy = correct_predictions / P * 100;


    end

    % Plot the result
    figure
    subplot(3, 1, 1)
    plot(cost_history)
    title('Cost history')
    xlabel('Iteration number')
    ylabel('Cost (g(w))')
    set(gca, 'XScale', 'log')
    set(gca, 'YScale', 'log')

    subplot(3, 1, 2)
    plot(w_history)
    title('Weight history')
    xlabel('Iteration number')
    ylabel('Weights')

    subplot(3, 1, 3)
    plot(accuracy)
    title('Accuracy history')
    xlabel('Iteration number')
    ylabel('Accuracy (%)')

end

