function [J, grad] = costfunctiontry(theta, X, Y, lambda)

  m = length(Y); 
  
  
  J = 0;
  grad = zeros(size(X,2),1);
   
  z = X * theta;      
  h_x = sigmoid(z);  
  
  reg_term = (lambda/(2*m)) * sum(theta(2:end).^2);
  
  J = (1/m)*sum((-Y.*log(h_x))-((1-Y).*log(1-h_x))) + reg_term; % scalar
  
  grad(1) = (1/m)* (X(:,1)'*(h_x-Y));                                  % 1 x 1
  grad(2:end) = (1/m)* (X(:,2:end)'*(h_x-Y))+(lambda/m)*theta(2:end);
end  