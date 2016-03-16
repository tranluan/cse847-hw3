% Function ridge_regressoin
%
% Luan Tran

function w_ml = ridge_regression(x,y, lambda)
    [U,S,V]=svd(x);
    eigval=diag(S);
    S_econ=diag(eigval);
    w_ml=V*inv(S_econ.^2+lambda*eye(size(S_econ)))*S'*U'*y;
end