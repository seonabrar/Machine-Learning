% Load the data from the file named '3class_data.csv'
A = load('3class_data.csv');

% Form the feature matrix starting from the first row up to the second last
X = A(1:end-1, :)';

% Form the class column vector from the last row in the data
y = A(end, :)';

% Number of classes
C = 3;

% Call the training function
W = trainOneVsAll(X, y, C);

function W = trainOneVsAll( X, y, C )

    % << IMPLEMENT FUNCTION BODY. SOME USUAL MAJOR STEPS ARE GIVEN IN THE COMMENTS BELOW >>

    % Initialize variables
    P = size(X, 1); 
    N = size(X, 2); 
    W = zeros(C, N + 1);
    
    % Perform One-vs-All: Train each class against all the others one by one
    for i = 1:C
        
        % Form the two-class problem
        %relabelling of y
         % Form the two-class problem
        new_label = (y == (i - 1)) * 2 - 1; % Convert class labels to +1/-1
        
        % Use trainPerceptron on the two-class problem
        [w, cost_history, w_history] = trainPerceptron(X, new_label);
        
        % Store the best weight
        [~, best_idx] = min(cost_history);
        W(i, :) = w_history(best_idx, :); % Pick the best weight here
        
        
    end

    % Normalize weights
    for i = 1:C
        Wc = W(i, 2:end); % Extract feature touching weights
        norm_Wc = sqrt(sum(Wc.^2)); % Compute L2 norm of feature touching weights
        W(i, :) = W(i, :) / norm_Wc; % Normalize the weight vector
    end

    % Check if perfect classification is achieved
    isPerfectClassification = all(classifyMultiClass(W, X) == y);
    disp(['Perfect classification of the sample data achieved: ' num2str(isPerfectClassification)]);
    
    % Count the number of misclassified samples
    misclassified = sum(classifyMultiClass(W, X) ~= y);
    disp(['The number of misclassified samples is: ' num2str(misclassified)]);
    
end