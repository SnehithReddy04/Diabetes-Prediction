%Logistic Regression


%loading data
data_set=readtable('C:\Users\snehith\Downloads\datasets_228_482_diabetes.csv', 'HeaderLines',1);
X=data_set(1:500,1:8);
Y=data_set(1:500,9);
[m, n] = size(X);
%making matrices
X=X{:,:};
Y=Y{:,:};
X = [ones(m, 1) X];





X_cv=data_set(501:650,1:8);
Y_cv=data_set(501:650,9);

X_test=data_set(651:768,1:8);
Y_test=data_set(651:768,9);

%making matrices
X_cv=X_cv{:,:};
Y_cv=Y_cv{:,:};
size(X_cv)
X_cv = [ones(150, 1) X_cv];

X_test=X_test{:,:};
Y_test=Y_test{:,:};
X_test = [ones(118, 1) X_test];


%scaling
for i =2:n+1
    me=mean(X(:,i));
    s=std(X(:,i));
    X(:,i)=((X(:,i)-me))/s;
end

for i =1:8
    me=mean(X_cv(:,i));
    s=std(X_cv(:,i));
    X_cv(:,i)=((X_cv(:,i)-me))/s;
end

for i =1:8
    me=mean(X_test(:,i));
    s=std(X_test(:,i));
    X_test(:,i)=((X_test(:,i)-me))/s;
end

%costfunction 
initial_theta=zeros(n+1,1);
lambda=1;

[cost, grad] = costfunctiontry(initial_theta, X, Y, lambda);

%using fminunc
%options = optimoptions(@fminunc,'Algorithm','Quasi-Newton','GradObj', 'on', 'MaxIter', 400);
%[theta,cost] = fminunc(@(t)(costfunctiontry(t, X, Y)), initial_theta, options);

options = optimset('GradObj', 'on', 'MaxIter', 500);
[theta,cost] = fmincg (@(t)(costfunctiontry(t, X, Y, lambda)), ...
                   initial_theta, options);
               
%cv error

 
               
%prediction
p=sigmoid(X*theta)>0.5;
acc=mean((double(p==Y))*100)


p_cv=sigmoid(X_cv*theta)>0.5;

acc_cv=mean((double(p_cv==Y_cv))*100)

p_test=sigmoid(X_test*theta)>0.5;

acc_cv=mean((double(p_test==Y_test))*100)


               


