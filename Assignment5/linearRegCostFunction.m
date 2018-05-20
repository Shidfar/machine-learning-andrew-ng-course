function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

%
%          1     m                      λ    n
% J(θ) = ----- { Σ (hₔ(xⁱ) - yⁱ)² } + ---- { Σ (θⱼ²) }
%         2m    i=1                    2m   j=i
%
%
%  ∂J(θ)     1    m                       λ
% ------- = --- { Σ (hₔ(xⁱ) - yⁱ)xⱼⁱ } + --- θⱼ
%   ∂θⱼ      m   i=1                      m
%

% z_theta = [0; theta];
z_theta = theta;
z_theta(1) = 0;

% s = num2cell(size(theta));
% printf("%d x %d\n", s{1}, s{2});
% s = num2cell(size(z_theta));
% printf("%d x %d\n", s{1}, s{2});

ht = X * theta;
err = ht - y;

J = (1 / (2*m)) * ( sum(err.^2) + ( lambda * sum(z_theta.^2) ) );
grad = (1 / m) * ((X' * err) + (lambda * z_theta));

% =========================================================================

grad = grad(:);

end
