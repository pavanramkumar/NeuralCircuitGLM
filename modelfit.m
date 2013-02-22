function [W,M,err] = modelfit(B,F,lambda2,method,NET)

switch(method)
  case 'L2'
  % Solves the underdetermined problem B = W*F by least-squares
  % regularization for an L2 norm cost function
  W = B*F'*pinv(F*F' + lambda2*eye(size(F,1)));
  M = eye(size(F,2));
  Bpred = W*F;
  %err = sqrt(mean(mean((B - Bpred).^2)));
  err = 1/(size(B,1)-size(B,2)-1)*mean(sum((B - Bpred).^2)./var(B));
  
  case 'IRLS'
  % Solves the underdetermined problem B = W*F by iterative reweighted least-squares
  % regularization for an Lp norm cost function
  TOL = 1e-2;
  MAXITER = 10;
  err = TOL+1;
  iter = 1;
  p = 1.1;
  M = eye(size(F,2));
  while ((err > TOL) && (iter < MAXITER))
    fprintf('  Iteration %02d  Error %f  M: [%f %f]\n',  iter, err, min(diag(M)), max(diag(M)));
    W = B*F'*pinv(F*M*F' + lambda2*eye(size(F,1)));
    M = diag(mean(power(abs(B - W*F), p-2)));     %Lp-norm
    M = M./sum(diag(M))*size(F,2);
    Bpred = W*F;
    %err = sqrt(mean(mean((B - Bpred).^2)));
    err = 1/(size(B,1)-size(B,2)-1)*mean(sum((B - Bpred).^2)./var(B));
    iter = iter+1;
  end

  case 'SVR'
  % Solves the underdetermined problem B = W*psi(F) by least-squares
  % regularization for an L2 norm cost function
  K = svmkernel(NET, F', F');
  %figure; imagesc(K); pause
  W = B*pinv(K + lambda2*eye(size(F,2)));
  M = eye(size(F,2));
  Bpred = W*svmkernel(NET, sqrt(M)*F', sqrt(M)*F');
  %err = sqrt(mean(mean((B - Bpred).^2)));
  err = 1/(size(B,1)-size(B,2)-1)*mean(sum((B - Bpred).^2)./var(B));
  
  case 'IRLSSVR'
  % Solves the underdetermined problem B = W*psi(F) by iterative reweighted least-squares
  % regularization for an Lp norm cost function
  TOL = 5e-1;
  MAXITER = 10;
  err = TOL+1;
  iter = 1;
  p = 1.1;
  M = eye(size(F,2));
  while ((err > TOL) && (iter < MAXITER))
    fprintf('  Iteration %02d  Error %f  M: [%f %f]\n',  iter, err, min(diag(M)), max(diag(M)));
    K = svmkernel(NET, sqrt(M)*F', sqrt(M)*F');
    W = B*pinv(K + lambda2*eye(size(F,2)));
    M = diag(mean(power(abs(B - W*svmkernel(NET, sqrt(M)*F', sqrt(M)*F')), p-2)));     %Lp-norm
    M = M./sum(diag(M))*size(F,2);
    Bpred = W*svmkernel(NET, sqrt(M)*F', sqrt(M)*F');
    %err = sqrt(mean(mean((B - Bpred).^2)));
    err = 1/(size(B,1)-size(B,2)-1)*mean(sum((B - Bpred).^2)./var(B));
    iter = iter+1;
  end

end
