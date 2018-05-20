function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
%
%   J(θ) = 1/m Σ(m, i=1)[-y[i]log(hθ(x[i])) - (1-y[i])log(1-hθ(x[i]))]
%
%   ∂J(θ)/∂θ[j] = 1/m Σ(m, i=1)[hθ(x[i]) − y[i]]x[i]
%

ht = sigmoid(X*theta);
left = ((-1.0)*y).*log(ht);
right = ((1.0)-y).*log(1-ht);

J = (1/m) * sum(left - right)

err = ht-y;
grad = (1.0/m)*(X'*err);

% =============================================================

end
