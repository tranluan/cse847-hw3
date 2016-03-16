
%load the data set
clear
load('diabetes.mat');

%% Training and test using all lambbas

%Append 1 at the dataset to introduce the bias term
x_train=[ones(size(x_train,1),1) x_train];
x_test=[ones(size(x_test,1),1) x_test];
X=x_train;
lambdas=[1e-5 1e-4 1e-3 1e-2 1e-1 1 10];


train_error = zeros(1,numel(lambdas));
test_error = zeros(1,numel(lambdas));


for i=1:numel(lambdas)
    w_ml = ridge_regression(x_train, y_train, lambdas(i));

    train_error(i)=sum((X*w_ml-y_train).^2)/size(y_train,1);
    test_error(i)=sum((x_test*w_ml-y_test).^2)/size(y_test,1);
end

figure;
hold on
plot(log10(lambdas),train_error,'r');
plot(log10(lambdas),test_error,'b');
xlabel('log_{10}(lambda)');
ylabel('MSE');

%% Performing 5-fold cross-validation on the training data
T=5;

numsamples=round(size(x_train,1)/T);
Xs=cell(T,1);
Ys=cell(T,1);
for i=1:T-1
   
    Xs{i}=x_train((i-1)*numsamples+1:numsamples*i,:);
    Ys{i}=y_train((i-1)*numsamples+1:numsamples*i);
end
Xs{T}=x_train((T-1)*numsamples+1:end,:);
Ys{T}=y_train((T-1)*numsamples+1:end);        

members=1:T;
L=1:T;
cv_error=zeros(T,numel(lambdas));

for i=1:T
   
    x_traincv=cat(1,Xs{members~=L(i)});
    y_traincv=cat(1,Ys{members~=L(i)});
    x_testcv=cat(1,Xs{L(i)});
    y_testcv=cat(1,Ys{L(i)});
    
    for j=1:length(lambdas)
        w_ml = ridge_regression(x_traincv, y_traincv, lambdas(j));
        cv_error(i,j)=sum((x_testcv*w_ml-y_testcv).^2)/size(y_testcv,1);
    end
end

% Best lambda 
[~,idx]=min(mean(cv_error,1));

% Plot
plot(log10(lambdas(idx))*ones(1,4001),[2000:6000],'g');
legend('Training error','Testing error','Best Lambda');
