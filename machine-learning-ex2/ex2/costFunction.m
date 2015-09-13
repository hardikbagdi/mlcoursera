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
%disp(X);
%disp(sigmoid(X));
%disp("This is thetea \n")
%disp(size(theta));
%transposey = y';


negativey =-y;
bagdians=log(sigmoid(X*theta));


%disp("size of bagdi ans:")
btranspose= bagdians';
%interm0=(log(sigmoid(X*theta)))';


interm1 = negativey.*bagdians;


%disp("This is size of interm1 \n")
%disp(size(interm1));
%disp("\n \n \n \n ++++++++++++++++++++++++++++++++");
%disp(log(1-sigmoid(X)));


interm2 = (1-y).*(log(1.-sigmoid(X*theta)));

%disp("This is size of interm2 \n")
%disp(size(interm2));


interm3 = interm1-interm2;


%disp("This is interm1\n \n ")
%disp(interm1);
%disp("This is interm2\n \n ")
%disp(interm2);
%disp("This is interm3")
%disp("This is size of interm3 \n")
%disp(size(interm3));
%disp(interm3);
%disp(sum(interm3));


J=(1/m).*sum(sum(interm3));


%disp(J);

% below for gradient


so2=(sigmoid(X*theta)-y)';
so3=[so2;so2;so2];
so1=X.*so3';
grad1=(sum(so1))';
grad=(1/m)*grad1;

% =============================================================

end
