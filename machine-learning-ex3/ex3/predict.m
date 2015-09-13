function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%


disp(size(X));
disp(size(Theta1));
disp(size(Theta2));

X=[ones(size(X,1),1) X];
for i=1:m
	
	curX=X(i,:);
	level1out=sum((curX.*Theta1),2);
	level2in=sigmoid(level1out);
	level2in=[1;level2in];
	level2out=level2in'.*Theta2;
	level3out=sigmoid(sum(level2out,2));
	[val,index]=max(level3out);
	p(i)=index;
endfor


% =========================================================================


end
