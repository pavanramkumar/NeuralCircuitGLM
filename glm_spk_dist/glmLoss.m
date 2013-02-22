
function [llhd, dx, H] = glmLoss(b,X,y,nu,offset,penalty)

if nargin<4, nu=0; end
if nargin<5, offset=[]; end
if nargin<6, penalty=b*0+1; end

lam = exp(X*b+offset);

% MLE
% llhd  = sum(lam) - sum(log(lam+(lam==0)).*y);
% dx = [X'*lam - X'*y];
% H  = ;

% MAP - L2
% llhd  = sum(lam) - sum(log(lam+(lam==0)).*y) + nu*norm(b.*penalty);
% dx = X'*lam - X'*y + nu*(b.*penalty)/norm(b.*penalty);

% MAP - L1
llhd  = sum(lam) - sum(log(lam+(lam==0)).*y) + nu*sum(abs(b.*penalty));
dx = X'*lam - X'*y + nu*sign(b.*penalty);

% llhd = nu*norm(b);
% dx = nu*b/norm(b);

% Call with...
% opts=optimset('Gradobj','on');
% fminunc(@glmLoss,[1 0 1]',opts,X,y,nu)