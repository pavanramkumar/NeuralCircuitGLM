
%% Simulate some presynaptic neurons...

clear Tlist
Npre=3;
params.dt = 0.01;
for i=1:Npre
    Tlist{i} = cumsum(exprnd(1,1000,1));
    Tlist{i} = Tlist{i}(Tlist{i}<2000);
end
S = getSpkMat(Tlist,params.dt,[],0);

%% Simulate postsynaptic neuron...

% Basis functions...
mprops.nfilt = 5;
mprops.delay = 200/(params.dt*1000);
mprops.basis = getBasis('rcos',mprops.nfilt,mprops.delay,20,0);
plot(mprops.basis')

% Covariates
X = getX(S,mprops.basis,0,1,0)';
B = randn(size(X,2)+1,1);
% [X,stobj] = stdize(X);
B(5:end)=0;
y = poissrnd(exp([X(:,1)*0+1 X]*B) * params.dt);

%% Fit..

% MLE
Bhat = glmfit(X,y,'poisson','offset',y*0+log(params.dt));

% MLE-Ian
opts=optimset('Gradobj','on');
[Xs,stobj]=stdize([X(:,1)*0+1 X]);
penalty = Bhat*0+1;
Bhati = fminunc(@glmLoss,Bhat,opts,Xs,y,0,y*0+log(params.dt),penalty);
Bhati = unstdize(Bhati,stobj);

% MAP(L1)-Ian
penalty = Bhat*0+1; penalty(1)=0;
Bhati2 = fminunc(@glmLoss,Bhat,opts,Xs,y,100,y*0+log(params.dt),penalty);
Bhati2 = unstdize(Bhati2,stobj);

[B Bhat Bhati Bhati2]


%%
for i=1:Npre
    subplot(Npre,1,i)
    plot(1:size(mprops.basis,2),exp(B((2:mprops.nfilt+1)+(i-1)*mprops.nfilt)'*mprops.basis),...
        1:size(mprops.basis,2),exp(Bhat((2:mprops.nfilt+1)+(i-1)*mprops.nfilt)'*mprops.basis),...
        1:size(mprops.basis,2),exp(Bhati2((2:mprops.nfilt+1)+(i-1)*mprops.nfilt)'*mprops.basis))
end