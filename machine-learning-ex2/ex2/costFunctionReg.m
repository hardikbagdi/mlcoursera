function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

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

FirstTerm= [0,0];
disp(FirstTerm);
%FirstTerm=costFunction(theta, X, y);
%disp(FirstTerm);
theta1 =theta(2:size(theta));
thetasquared= theta1.*theta1;
negativey =-y;
bagdians=log(sigmoid(X*theta));
interm1 = negativey.*bagdians;
interm2 = (1-y).*(log(1.-sigmoid(X*theta)));
interm3 = interm1-interm2;
interm4= (lambda/(2*m))*sum(thetasquared);
J=((1/m).*sum(sum(interm3)))+interm4;
%disp("This is J");
%disp(J);
% =============================================================

so2=(sigmoid(X*theta)-y)';
so3=[so2;so2;so2];

so1=X.*so3';
grad1=(sum(so1))';

grad1=(1/m)*grad1;
zeroth=grad1(1);
grad1=grad1+((lambda/m).*theta);
grad1(1)=zeroth;
grad=grad1;




% =============================================================

end