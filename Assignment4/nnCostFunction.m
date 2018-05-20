function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
% -------------------------------------------------------------


                        % % X -> 5000x400
a1 = [ones(m,1) X];     % % a1 -> 5000x401

                        % % Theta1 -> 25x401
                        % % Theta2 -> 10x26
                        % % y -> 5000x1

z2 = a1 * Theta1';
a2 = sigmoid(z2);
a2 = [ones(m, 1) a2];
z3 = a2 * Theta2';
a3 = sigmoid(z3);

ht = a3;

% %
% repmat:
%    https://octave.sourceforge.io/octave/function/repmat.html
%    https://octave.org/doc/v4.2.2/Special-Utility-Matrices.html
%    repmat (A, m, n):
%         Form a block matrix of size m by n, with a copy of matrix A as each element.
% %

check = repmat([1:num_labels], m, 1);  % 1...10 repeat `m` times (m rows -> 5000x10)
matrix_y = repmat(y, 1, num_labels);   % y repeat `num_labels` times (num_labels columns -> 5000x10)
y = check == matrix_y;  % returns a matrix with ones where there is a match for y[i] and zeros otherwise (5000x10)

% %
%
% s = num2cell(size(check));
% printf("%d x %d\n", s{1}, s{2});
% s = num2cell(size(matrix_y));
% printf("%d x %d\n", s{1}, s{2});
% s = num2cell(size(y));
% printf("%d x %d\n", s{1}, s{2});
%
% %

left = y.*log(ht);
right = (1 - y).*log(1 - ht);
cost = (-1 / m)*sum(sum(left + right));

regTheta1 = Theta1(:, 2: end);
regTheta2 = Theta2(:, 2: end);

regularization = (lambda / (2*m)) * (sum(sum(regTheta1.^2)) + sum(sum(regTheta2.^2)));
J = cost + regularization;

D1 = zeros(size(Theta1));
D2 = zeros(size(Theta2));

for t = 1:m

%     z2 = a1 * Theta1';
%     a2 = sigmoid(z2);
%     a2 = [ones(m, 1) a2];
%     z3 = a2 * Theta2';
%     a3 = sigmoid(z3);
    a1t = a1(t, :);
    z2t = [1; Theta1 * a1t'];
    a2t = a2(t, :);
    a3t = a3(t, :);
    yt = y(t, :);

%     δ(2) = 􏰀Θ(2)􏰁T δ(3). ∗ g′(z(2))
    d3 = a3t - yt;
    d2 = Theta2' * d3' .* sigmoidGradient(z2t);

%     ∆(l) = ∆(l) + δ(l+1)(a(l))T
    D2 = D2 + d3' * a2t;
    D1 = D1 + d2(2:end) * a1t;

end

reg1 = [zeros(size(Theta1, 1), 1) regTheta1];   %  [0][ The   ]
reg2 = [zeros(size(Theta2, 1), 1) regTheta2];   %  [0][ ta(l) ]

Theta1_grad = 1/m * D1 + (lambda/m) * reg1;
Theta2_grad = 1/m * D2 + (lambda/m) * reg2;


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end