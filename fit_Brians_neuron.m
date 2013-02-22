%load fake_Neuron.mat

BINSIZE = 100;
spiketimes = find(fake_neur == 1);
spikes = histc(spiketimes, [1:BINSIZE:length(fake_neur)]);
thetavec = decimate(angs, BINSIZE);
dirvec = [cos(thetavec) sin(thetavec)];

timedur = 4001:5000;
X_GLM = dirvec(timedur, :);
X_GKM = thetavec(timedur);

Y = spikes(timedur);

% GLM fitting
[Bhat, dev, stats] = glmfit(X_GLM, Y,'poisson');
Yhat_GLM = exp([ones(1000,1), X_GLM]*Bhat);
R2_GLM = compute_pseudo_R2(Y, Yhat_GLM);

% GKM fitting
net = kpor('kernel', rbf('eta', 0.1), 'lambda', 0.1, 'Verbosity', 'ethereal');
selector  = simplex('estimator', aloo, 'TolFun', 1e-6, 'TolX', 1e-6);
net = select(selector, net, X_GKM, Y);
Yhat_GKM = fwd(net, X_GKM);
R2_GKM = compute_pseudo_R2(Y, Yhat_GKM);

% Visualize fits
figure;
hold on; plot(thetavec(timedur), Yhat_GLM, 'kx');
hold on; plot(thetavec(timedur), Yhat_GKM, 'rx');
%hold on; plot(thetavec(timedur), Y, 'gx');

%hold on; plot((1:361)-181, tunecurve, 'b');
legend(sprintf('GLM R2 = %f', R2_GLM), sprintf('GKM R2 = %f', R2_GKM));
title('Simulated neuron');
xlabel('Direction (degrees)');
ylabel('Normalized firing rate');

