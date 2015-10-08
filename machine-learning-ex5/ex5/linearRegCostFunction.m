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

%display(size(theta));
%display(X);

differencebagdi= X*theta;
%display(differencebagdi);
insidesum=(differencebagdi-y).^2;
summation=sum(insidesum);
%disp(summation);
denom=2*m;
bagdi=summation/denom;
%disp('Sum is bagdo:');disp(bagdi);

filteredtheta = theta(2:size(theta));
filteredtheta = filteredtheta.*filteredtheta;
regularizationterm= (lambda/(denom)).*(sum(filteredtheta));
J=bagdi+regularizationterm;


gradient1 =((1/m) .* X'*(X*theta - y));
gradient2= (lambda/m).*theta;
gradient2(1)=0;
grad=gradient1+gradient2;






% =========================================================================

grad = grad(:);

end
