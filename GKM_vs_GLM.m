%% Read in the file and sync neural and kinematic streams
load ../MrT_data_9_24_2012.mat
load ../trials2.mat

y = [];
speedvec_lo = []; speedvec_hi = [];
dirvec_lo = []; dirvec_hi = []; 
thetavec_lo = []; thetavec_hi = [];

spikevec_lo = cell(length(trains_PMd),1);
spikevec_hi = cell(length(trains_PMd),1);

BINSIZE = 50;

% Extract relevant timestamps
for tr=1:length(trials2) 
    fprintf('Trial %d\n', tr);
    % First cut out the spikes for each neuron
    xshift(tr) = trials2(tr,1);
    trialstart(tr) = 1000*trials2(tr,3);
    feedback(tr) = 1000*trials2(tr,4);
    trialend(tr) = 1000*trials2(tr,5);
end

%%
for tr=1:length(trials2)
    START = feedback(tr);
    END = trialend(tr);
    for n=1:length(trains_PMd)
        %fprintf('    Neuron %d\n', n);
        spiketimes = 1000*trains_PMd{n};
        trial_spiketimes = spiketimes(spiketimes >= feedback(tr) & spiketimes <= trialend(tr));
        
        if(isempty(trial_spiketimes))
          spikes = zeros(length([feedback(tr):BINSIZE:trialend(tr)]), 1);
        else
          spikes = histc(trial_spiketimes, [round(feedback(tr)):BINSIZE:round(trialend(tr))]);
        end
        if(size(spikes,1) < size(spikes,2))
            spikes = spikes';
            %fprintf('Trial %d Neuron %d is weird\n', tr, n); 
        end
        if(trials2(tr,2) == 0.5)
            spikevec_lo{n} = [spikevec_lo{n}; spikes];
        else
            spikevec_hi{n} = [spikevec_hi{n}; spikes];
        end
    end
    
    % Then select the velocity traces
    trial_vel = pva.vel((1000*pva.vel(:,1) >= feedback(tr) & ...
                         1000*pva.vel(:,1) <= trialend(tr)),2:3);
    clear trial_vel_ds;
    trial_vel_ds(:,1) = decimate(trial_vel(:,1), BINSIZE);
    trial_vel_ds(:,2) = decimate(trial_vel(:,2), BINSIZE);
    
    if(length(trial_vel_ds) > length(spikes))
        %fprintf('\t \t Realigning 1!!\n');
        trial_vel_ds(length(spikes)+1:end,:) = [];
    end
    if(length(spikes) > length(trial_vel_ds))
        trial_vel_ds = [trial_vel_ds; repmat(trial_vel_ds(end,:), [length(spikes) - length(trial_vel_ds) 1])];
        %fprintf('\t \t Realigning 2!! %d %d\n', length(spikes), length(trial_vel_ds));
    end
    
    trial_vel_ds = trial_vel_ds./repmat(sqrt(sum(trial_vel_ds.^2, 2)), [1 2]);
    theta = 180/pi*atan2(trial_vel_ds(:,2), trial_vel_ds(:,1));
    if(trials2(tr,2) == 0.5)
        speedvec_lo = [speedvec_lo; sqrt(sum(trial_vel_ds.^2, 2))];
        dirvec_lo = [dirvec_lo; [cos(theta*pi/180) sin(theta*pi/180)]];
        thetavec_lo = [thetavec_lo; theta];
    else
        speedvec_hi = [speedvec_hi; sqrt(sum(trial_vel_ds.^2, 2))];
        dirvec_hi = [dirvec_hi; [cos(theta*pi/180) sin(theta*pi/180)]];
        thetavec_hi = [thetavec_hi; theta];
    end
    
    fprintf('Trial %d\n', tr);
end

% Collect spike trains of all neurons in a single matrix
y_lo = horzcat(spikevec_lo{:,:})/BINSIZE*1000;
y_hi = horzcat(spikevec_hi{:,:})/BINSIZE*1000;


%%
% Simulate a neuron with known tuning curve
tunecurve = zeros(361,1);
th = -180:180;
%tunecurve(181:end) = gausswin(181);
a1 = 0.3;
a2 = 0.3;
a3 = sqrt(1-a1^2-a2^2);
tunecurve = a1*cos(th/180*pi)' + ...
            a2*cos(2*th/180*pi)' + ...
            a3*cos(3*th/180*pi)';
lambda = exp(tunecurve(round(thetavec_lo+181)));
simulated_spikes = poissrnd(lambda);


% Apply the GLM
n = 6;
timedur = 2001:3000;
X_GLM_lo = dirvec_lo(timedur, :);
X_GLM_hi = dirvec_hi(timedur, :);
X_GKM_lo = thetavec_lo(timedur);
X_GKM_hi = thetavec_hi(timedur);

Y_lo = y_lo(timedur,n);
Y_hi = y_hi(timedur,n);
Y_sim = simulated_spikes(timedur);

[Bhat, dev, stats] = glmfit(X_GLM_lo, Y_lo,'poisson');
Yhat_GLM_lo = exp([ones(1000,1), X_GLM_lo]*Bhat);
R2_GLM_lo = compute_pseudo_R2(Y_lo, Yhat_GLM_lo);

[Bhat, dev, stats] = glmfit(X_GLM_hi, Y_hi,'poisson');
Yhat_GLM_hi = exp([ones(1000,1), X_GLM_hi]*Bhat);
R2_GLM_hi = compute_pseudo_R2(Y_hi, Yhat_GLM_hi);

[Bhat, dev, stats] = glmfit(X_GLM_lo, Y_sim,'poisson');
Yhat_GLM_sim = exp([ones(1000,1), X_GLM_lo]*Bhat);
R2_GLM_sim = compute_pseudo_R2(Y_sim, Yhat_GLM_sim);

% Apply GKM
%fix(gkm('acronym', 'kpor', ...
%        'name',    'kernel Poisson regression', ...
%        'canonical',  'exp(eta)'));

net = kpor('kernel', rbf('eta', 1e-5), 'lambda', 0.1, 'Verbosity', 'ethereal');
selector  = simplex('estimator', aloo, 'TolFun', 1e-6, 'TolX', 1e-6);
net = select(selector, net, X_GKM_lo, Y_lo);
Yhat_GKM_lo = fwd(net, X_GKM_lo);
R2_GKM_lo = compute_pseudo_R2(Y_lo, Yhat_GKM_lo);

net = kpor('kernel', rbf('eta', 1e-5), 'lambda', 0.1, 'Verbosity', 'ethereal');
selector  = simplex('estimator', aloo, 'TolFun', 1e-6, 'TolX', 1e-6);
net = select(selector, net, X_GKM_hi, Y_hi);
Yhat_GKM_hi = fwd(net, X_GKM_hi);
R2_GKM_hi = compute_pseudo_R2(Y_hi, Yhat_GKM_hi);

net = kpor('kernel', rbf('eta', 1e-2), 'lambda', 0.1, 'Verbosity', 'ethereal');
selector  = simplex('estimator', aloo, 'TolFun', 1e-6, 'TolX', 1e-6);
net = select(selector, net, X_GKM_lo, Y_sim);
Yhat_GKM_sim = fwd(net, X_GKM_lo);
R2_GKM_sim = compute_pseudo_R2(Y_lo, Yhat_GKM_sim);

% Compare GLM vs GKM by plotting tuning curve
% Real neuron
figure; subplot(1,2,1);
hold on; plot(thetavec_lo(timedur), Yhat_GKM_lo, 'kx');
hold on; plot(thetavec_hi(timedur), Yhat_GKM_hi, 'rx');
legend(sprintf('Lo Unc R2 = %f', R2_GKM_lo), sprintf('Hi Unc R2 = %f', R2_GKM_hi));
%hold on; plot(thetavec_lo(timedur), Y_lo, 'ko');
%hold on; plot(thetavec_hi(timedur), Y_hi, 'ro');
title('Neuron 6: GKM predicted tuning');
xlabel('Direction (degrees)');
ylabel('Firing rate (spikes/s)');

subplot(1,2,2);
hold on; plot(thetavec_lo(timedur), Yhat_GLM_lo, 'kx');
hold on; plot(thetavec_hi(timedur), Yhat_GLM_hi, 'rx');
legend(sprintf('Lo Unc R2 = %f', R2_GLM_lo), sprintf('Hi Unc R2 = %f', R2_GLM_hi));
%hold on; plot(thetavec_lo(timedur), Y_lo, 'ko');
%hold on; plot(thetavec_hi(timedur), Y_hi, 'ro');
title('Neuron 6: GLM predicted tuning');
xlabel('Direction (degrees)');
ylabel('Firing rate (spikes/s)');

% Simulated neuron
figure;
hold on; plot(thetavec_lo(timedur), Yhat_GLM_sim, 'kx');
hold on; plot(thetavec_lo(timedur), Yhat_GKM_sim, 'rx');
hold on; plot(thetavec_lo(timedur), Y_sim, 'gx');

hold on; plot((1:361)-181, tunecurve, 'b');
legend(sprintf('GLM R2 = %f', R2_GLM_sim), sprintf('GKM R2 = %f', R2_GKM_sim), 'Tuning curve');
title('Simulated neuron');
xlabel('Direction (degrees)');
ylabel('Normalized firing rate');

