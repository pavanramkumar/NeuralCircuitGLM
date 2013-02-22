% Create an estimate a simluated spiking network
close all
clear all
%% Create network

% Design a basis (this is unknown to the estimator)
BINSIZE = 1;
addpath('glm_spk_dist')
P = 5;
delay = 50*BINSIZE;

%[t, ~, HiddenBasis] = makeRaisedCosBasis(P, BINSIZE/1000, [3/1000 20/1000], 1.5*1e-2, 1);
% HiddenBasis(:,1) = [];

HiddenBasis = getBasis('rcos',P,delay,20,0)';
t = linspace(1,delay,length(HiddenBasis));

% Number of neurons
N = 6;
% Length of spiketrain (ms)
L = 200000;

% Randomly generate a set of kernel functions
% Load a pre-designed set of basis weights
%load KernelWs
figure(11)
alph = [];
for n1=1:N
    tmp = [];
    for n2=1:N
        if(n1 == n2)
            W(n1,n2,:) = randn(P,1)-1;    % Make sure the first basis function is negatively weighted, to ensure refractory period
            W(n1,n2,1) = W(n1,n2,1)-5;
        else
            W(n1,n2,:) = 0.5*randn(P,1);
        end
        K(n1,n2,:) = HiddenBasis*squeeze(W(n1,n2,:));
        tmp = [tmp, squeeze(K(n1,n2,:))'];
        subplot(N,N,N*(n1-1)+n2); plot(t, exp(squeeze(K(n1,n2,:))), 'k', 'LineWidth', 2);
    end
    alph = [alph; tmp];
end

% Set the baseline firing rates
% baseline = 0.2*ones(N,1);
baseline = -1-rand(N,1);

% External covariates

% Simulate a population with Cosine directional tuning
thetavec = 2*pi*smooth(rand(L+size(K,3)+20,1)-0.5, 1);
thetavec = thetavec(11:end-10);
X = [cos(thetavec) sin(thetavec)];
theta_0 = pi/4;
%a1 = 0.4;
%a2 = sqrt(1-a1^2);
a1 = cos(theta_0);
a2 = sin(theta_0);

beta = [a1; a2];
figure(12); plot(thetavec, exp(X*beta), 'k.');

% X = [];
% beta = [];

%% Simulate the network (time-rescaling by Brown et al. 2001)

% My function
% spiketrains = simulateNetwork(baseline, K, L, beta, X);

% spiketrains = poissrnd(exp(baseline + X*beta))';

% Ian's function
alph = [baseline, alph];
firings = simLNP_trsX(alph, 1, L+size(K,3), beta', X);
spiketrains = zeros(N, L+size(K,3));
for n=1:N
    spiketrains(n,firings(firings(:,2) == n, 1)) = 1;
end

% Plot simulated spikes
figure(13); hold on;
for n=1:N
    plot(find(spiketrains(n,:) == 1), n, 'k.');
end
ylim([0 6])

%% Estimate network using GLM
% GLMs for functional connectivity
% One GLM per neuron with lagged version of other neurons as predictors

% Design a basis (using the same as the hidden basis, for now)
BINSIZE = 1;
addpath('glm_spk_dist')
P = 5;
delay = 50*BINSIZE;

%[t, ~, Basis] = makeRaisedCosBasis(P, BINSIZE/1000, [3/1000 20/1000], 1.5*1e-2, 1);
% Basis(:,1) = [];

Basis = getBasis('rcos',P,delay,20,0)';
t = linspace(1,delay,length(Basis));

%[t, Basis] = makeDataDrivenBasis(spiketrains, size(K,3));

% Collect all the spiketrains
Xfc = spiketrains';

ConMat = [];

fprintf('Computing delayed version of all neurons...');
X_filtered = [];
for p=1:N
    for b=1:size(Basis,2)
        X_filtered = [X_filtered, filter(Basis(:,b), 1, Xfc(:,p))];
    end
end
fprintf('[done]\n');

% Also add external covariates if they exist
if(~isempty(beta))
    X_filtered = [X_filtered, cos(thetavec), sin(thetavec)];
end

nbasis = size(Basis,2);
for n=1:N
    YfcTrain = Xfc(1:L/2,n);
    YfcTest = Xfc(L/2+1:end,n);
    XfcTrain = X_filtered(1:L/2,:);
    XfcTest = X_filtered(L/2+1:end,:);
    
    % Fit GLMs
    fprintf('\tFitting GLMs...');
    
    % Ridge regression
    [B, dev, stats] = glmfit(XfcTrain, YfcTrain, 'poisson');
    
    % Lasso/ Elastic Net regression
    %[Blasso, Slasso] = lassoglm(Xfc, Yfc, 'poisson', 'alpha', 0.8, 'cv', 10);
    %B = [Slasso.Intercept(Slasso.IndexMinDeviance); Blasso(:,Slasso.IndexMinDeviance)];
    
    fprintf('[done]\n');
    YfcTrain_hat = exp([ones(length(XfcTrain),1), XfcTrain]*B);
    TrainR2 = compute_pseudo_R2(YfcTrain, YfcTrain_hat);
    
    YfcTest_hat = exp([ones(length(XfcTest),1), XfcTest]*B);
    TestR2 = compute_pseudo_R2(YfcTest, YfcTest_hat);
    
    fprintf('\tTrainR2: %6.4f TestR2: %6.4f\n', TrainR2, TestR2);
    
    spiketrains_hat(n,:) = [YfcTrain_hat', YfcTest_hat'];
    
    % Collect the coefficients 
    ConMat(n,:) = B';
end

% Visualize estimated kernels
figure(11); hold on;
for n1=1:N
    for n2=1:N
        subplot(N,N,(n1-1)*N+n2); hold on;
        plot(t, exp(Basis*ConMat(n1,P*(n2-1)+2:P*n2+1)'), 'r', 'LineWidth', 2);
        %ylim([0 1.5]);
        %axis('off'); box('off');
    end
end

% Visualize predicted spike trains
figure(13); hold on;
for n=1:N
    plot(n + 0.4*spiketrains_hat(n,:), 'r');
end

% Visualized predicted tuning function
figure(12); hold on;
PredTune = (ConMat(:,[P*N+2 P*N+3])*[cos(thetavec) sin(thetavec)]')';
for n=1:N
    plot(thetavec,  exp(PredTune(:,n)), 'r.');
end

% Recover preferred direction for each neuron
b1hat = ConMat(:,P*N+2);
b2hat = ConMat(:,P*N+3);
for n=1:N
   theta_0_hat(n) = atan2(b2hat(n), b1hat(n));
end
theta_0_hat
