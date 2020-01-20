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

X = [ones(m,1), X];
Y = (1:num_labels) == y;

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

% Part 1

for i = 1:m
    h = sigmoid(Theta2 * [1; sigmoid(Theta1 * X(i,:)')]); % 10x1
    for k = 1:num_labels
        J = J + (1/m)*(-Y(i,k)*log(h(k)) - (1 - Y(i,k))*log(1 - h(k)));
    end
end

sum1 = 0;
for j = 1:length(Theta1(:,1))
    for k = 2:length(Theta1(1,:))
        sum1 = sum1 + Theta1(j,k)^2;
    end
end

sum2 = 0;
for j = 1:length(Theta2(:,1))
    for k = 2:length(Theta2(1,:))
        sum2 = sum2 + Theta2(j,k)^2;
    end
end

reg = (lambda/(2*m))*(sum1 + sum2);

J = J + reg;


% Part 2 
Delta1 = 0;
Delta2 = 0;

for i = 1:m
    z2 = Theta1 * X(i,:)'; % 25x1
    a2 = [1; sigmoid(z2)]; % 26x1
    z3 = Theta2 * a2; % 10x1
    a3 = sigmoid(z3); % 10x1
    delta3 = a3 - Y(i,:)'; % 10x1
    delta2 = Theta2'*delta3; % 26x1
    delta2 = delta2(2:end) .* sigmoidGradient(z2); % 25x1
    
    Delta1 = Delta1 + delta2*X(i,:); % 25x401
    Delta2 = Delta2 + delta3*a2'; % 10x26
end

Theta1_grad = Delta1/m;
Theta2_grad = Delta2/m;

% Part 3
for i = 1:length(Theta1_grad(:,1))
    for j = 2:length(Theta1_grad(1,:))
        Theta1_grad(i,j) = Theta1_grad(i,j) + lambda*Theta1(i,j)/m;
    end
end

for i = 1:length(Theta2_grad(:,1))
    for j = 2:length(Theta2_grad(1,:))
        Theta2_grad(i,j) = Theta2_grad(i,j) + lambda*Theta2(i,j)/m;
    end
end
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
