%% Read in the file and sync neural and kinematic streams

load ../MrT_data_01232013.mat
% load ../trials2.mat

trains_PMd = PMd_units; clear PMd_units
trains_M1 = M1_units; clear M1_units
LOval = 1.5; HIval = 3.0;

BINSIZE = 10;

%% Extract time courses
% Kinematics
speedvec = sqrt(pva.vel(:,2).^2 + pva.vel(:,2).^2);
thetavec = atan2(pva.vel(:,3), pva.vel(:,2));

speedvec = decimate(speedvec, BINSIZE);
thetavec = decimate(thetavec, BINSIZE);

dirvec = [cos(thetavec) sin(thetavec)];

% Neurons
spikevec_PMd = [];
spikevec_M1 = [];

for n=1:length(trains_PMd)
    fprintf('    Neuron %d\n', n);
    spiketimes = 1000*trains_PMd{n};
    spikes = histc(spiketimes, [1000*pva.vel(1,1):BINSIZE:1000*pva.vel(end,1)]);
    spikevec_PMd(:,n) = spikes;
end
for n=1:length(trains_M1)
    fprintf('    Neuron %d\n', n);
    spiketimes = 1000*trains_M1{n};
    spikes = histc(spiketimes, [1000*pva.vel(1,1):BINSIZE:1000*pva.vel(end,1)]);
    spikevec_M1(:,n) = spikes;
end

% Uncertainty
LOvec = zeros(length(thetavec),1);
HIvec = zeros(length(thetavec),1);

for tr=1:length(trials) 
    fprintf('Trial %d\n', tr);
    % First cut out the spikes for each neuron
    xshift(tr) = trials(tr,1);
    trialstart(tr) = 1000*trials(tr,3)-999;
    feedback(tr) = 1000*trials(tr,4)-999;
    trialend(tr) = 1000*trials(tr,5)-999;
    index = find(1000*pva.pos(:,1) > trialend(tr), 1, 'first');
    endpoint(tr) = pva.pos(index,2);

    if(trials(tr,2) == LOval)
        LOvec(round(trialstart(tr)/BINSIZE):round(trialend(tr)/BINSIZE)) = 1;
    else
        HIvec(round(trialstart(tr)/BINSIZE):round(trialend(tr)/BINSIZE)) = 1;
    end
end

%% ---------------------------------------------
%   Plot PSTHs for LO and HI unc trials
%-----------------------------------------------
trLO = find(trials(:,2) == LOval);
trHI = find(trials(:,2) == HIval);

% Separate into left and right reaches
LO_L = (endpoint(trLO) <= -1);
LO_R = (endpoint(trLO) > -0.5);
HI_L = (endpoint(trHI) <= -1);
HI_R = (endpoint(trHI) > -0.5);

PSTH_t = (0:299)/100;
clear PSTH_LO PSTH_HI
spikevec = [spikevec_M1, spikevec_PMd];
NeuronSelect_M1 = find(sum(spikevec_M1,1) > 10000);
NeuronSelect_PMd = find(sum(spikevec_PMd,1) > 10000);

addpath('boundedline');

plotidx = 1; figure
for n =1:110
  S = [];
  for tst=1:length(trLO)
    S = [S; spikevec(round(trialstart(trLO(tst))/BINSIZE):round(trialstart(trLO(tst))/BINSIZE)+299, n)'];
  end
  PSTH_LO(:,:,n) = S/BINSIZE*1000;
  
  S = [];
  for tst=1:length(trHI)
    S = [S; spikevec(round(trialstart(trHI(tst))/BINSIZE):round(trialstart(trHI(tst))/BINSIZE)+299, n)'];
  end
  PSTH_HI(:,:,n) = S/BINSIZE*1000;
  
  if((n <= 64 && ~isempty(find(NeuronSelect_M1 == n, 1))) || (n > 64 && ~isempty(find(NeuronSelect_PMd == n-64, 1))))
    subplot(6,7,plotidx); hold on;
    boundedline(PSTH_t, smooth(squeeze(mean(PSTH_LO(LO_L,:,n),1))), squeeze(std(PSTH_LO(LO_L,:,n),[],1))/sqrt(sum(LO_L)), 'r', ...
                PSTH_t, smooth(squeeze(mean(PSTH_LO(LO_R,:,n),1))), squeeze(std(PSTH_LO(LO_R,:,n),[],1))/sqrt(sum(LO_R)), 'g', ...
                PSTH_t, smooth(squeeze(mean(PSTH_HI(HI_L,:,n),1))), squeeze(std(PSTH_HI(HI_L,:,n),[],1))/sqrt(sum(HI_L)), 'b', ...
                PSTH_t, smooth(squeeze(mean(PSTH_HI(HI_R,:,n),1))), squeeze(std(PSTH_HI(HI_R,:,n),[],1))/sqrt(sum(HI_R)), 'k');
  
    xlim([0 0.8]); 
    %ylim([0 60]);
    plotidx = plotidx + 1;
  end
  
  if(n <= 64)
    title(sprintf('M1 neuron: %d', n));
  else
    title(sprintf('PMd neuron: %d', n-64));
  end
end

%% ---------------------------------------------
%   Decode UNC from PSTHs
%-----------------------------------------------
Success = decode_uncertainty_4class(PSTH_LO, PSTH_HI, LO_L, LO_R, HI_L, HI_R, [30:80], [NeuronSelect_M1, NeuronSelect_PMd+64], 'forests');


%% Design temporal basis functions and convolve them
addpath('glm_spk_dist')
%[iht, ihbas, ihbasis] = makeRaisedCosBasis(3, BINSIZE/1000, [2/1000 25/1000], 1.5*1e-2, 1);
%Basis = ihbasis;

Basis = getBasis('rcos',3,10,4,0)';
iht = linspace(1,length(Basis),length(Basis));

%load Basis

% Kinematics must follow neural activity
% So push it backward in time by filtering in reverse
temp = filter(Basis(:,1), 1, dirvec(end:-1:1,1)); X(:,1) = temp(end:-1:1);
temp = filter(Basis(:,2), 1, dirvec(end:-1:1,1)); X(:,2) = temp(end:-1:1);
temp = filter(Basis(:,3), 1, dirvec(end:-1:1,1)); X(:,3) = temp(end:-1:1);

temp = filter(Basis(:,1), 1, dirvec(end:-1:1,2)); X(:,4) = temp(end:-1:1);
temp = filter(Basis(:,2), 1, dirvec(end:-1:1,2)); X(:,5) = temp(end:-1:1);
temp = filter(Basis(:,3), 1, dirvec(end:-1:1,2)); X(:,6) = temp(end:-1:1);

temp = filter(Basis(:,1), 1, speedvec(end:-1:1,1)); X(:,7) = temp(end:-1:1);
temp = filter(Basis(:,2), 1, speedvec(end:-1:1,1)); X(:,8) = temp(end:-1:1);
temp = filter(Basis(:,3), 1, speedvec(end:-1:1,1)); X(:,9) = temp(end:-1:1);

Xnames = {'Cos1', 'Cos2', 'Cos3', 'Sin1', 'Sin2', 'Sin3', 'Speed1', 'Speed2', 'Speed3', 'LowUnc1', 'LowUnc2', 'LowUnc3', 'HighUnc1', 'HighUnc2', 'HighUnc3'};

% Uncertainty must lead neural activity
% So push it forward in time by filtering normally
X(:,10) = filter(Basis(:,1), 1, LOvec);
X(:,11) = filter(Basis(:,2), 1, LOvec);
X(:,12) = filter(Basis(:,3), 1, LOvec);
X(:,13) = filter(Basis(:,1), 1, HIvec);
X(:,14) = filter(Basis(:,2), 1, HIvec);
X(:,15) = filter(Basis(:,3), 1, HIvec);

%% GLMs for functional connectivity
% One GLM per neuron
% 14 neurons x 3 basis functions = 42
% 5 kinematic terms x 5 basis functions = 25

[~, ihbas, ihbasis] = makeRaisedCosBasis(5, BINSIZE/1000, [2/1000 25/1000]*7, 1*1.5*1e-2, 1);
KinBasis = ihbasis;
BASIS{1} = KinBasis; BASIS{2} = KinBasis; BASIS{3} = KinBasis; BASIS{4} = KinBasis; BASIS{5} = KinBasis;

[~,Xkin,Xkin_names,trial_inds]=Do_Covariates(pva,trials,BINSIZE,1,BASIS);
clc
% Select neurons which fire > 10000 spikes

NeuronSelect_M1 = find(sum(spikevec_M1,1) > 10000);
NeuronSelect_PMd = find(sum(spikevec_PMd,1) > 10000);
L1 = length(NeuronSelect_M1);
L2 = length(NeuronSelect_PMd);
L = L1+L2;
Xneurons = [spikevec_M1(:,NeuronSelect_M1), spikevec_PMd(:,NeuronSelect_PMd)];
ConMat_LO = []; ConMat_HI = []; ConMat = [];

fprintf('Computing delayed version of all neurons...');
Xneurons_filtered = [];
for p=1:size(Xneurons,2)
    for b=1:size(Basis,2)
        Xneurons_filtered = [Xneurons_filtered, filter(Basis(:,b), 1, Xneurons(:,p))];
    end
end
fprintf('[done]\n');

%% ---------------------------------------------
%   Fit GLMs with 10-fold CV
%-----------------------------------------------
trLO = find(trials(:,2) == LOval);
trHI = find(trials(:,2) == HIval);

tmpLO = trLO(randperm(length(trLO)));
tmpHI = trHI(randperm(length(trHI)));

global R2r R2t

setenv('DYLD_LIBRARY_PATH', '/usr/local/bin');

NCV = 2;
for cv=1:NCV
    fprintf('---------------\n');
    fprintf('CV fold: %d\n', cv);
    fprintf('---------------\n');
    
    testLO = tmpLO((cv-1)*100+1:cv*100);
    trainLO = setdiff(trLO, testLO, 'stable');
    testHI = tmpHI((cv-1)*100+1:cv*100);
    trainHI = setdiff(trHI, testHI, 'stable');
    TrainSELECT_LO = []; TestSELECT_LO = [];
    TrainSELECT_HI = []; TestSELECT_HI = [];
    for tr1=1:length(trainLO)
        TrainSELECT_LO = [TrainSELECT_LO, round(trialstart(trainLO(tr1))/BINSIZE):round(trialend(trainLO(tr1))/BINSIZE)];
    end
    for tr1=1:length(testLO)
        TestSELECT_LO = [TestSELECT_LO, round(trialstart(testLO(tr1))/BINSIZE):round(trialend(testLO(tr1))/BINSIZE)];
    end
    for tr1=1:length(trainHI)
        TrainSELECT_HI = [TrainSELECT_HI, round(trialstart(trainHI(tr1))/BINSIZE):round(trialend(trainHI(tr1))/BINSIZE)];
    end
    for tr1=1:length(testHI)
        TestSELECT_HI = [TestSELECT_HI, round(trialstart(testHI(tr1))/BINSIZE):round(trialend(testHI(tr1))/BINSIZE)];
    end
    
    for n=1:L
        if(n <= L1)
            fprintf('M1 Neuron %d\n', NeuronSelect_M1(n));
        else
            fprintf('PMd Neuron %d\n', NeuronSelect_PMd(n-L1));
        end

        Yfc = Xneurons(:,n);
    
        % Delay all the other neurons with basis functions
        fprintf('\tComputing delayed version of other neurons...');
        Xfc = Xneurons_filtered;
        %Xfc(:,3*(n-1)+1:3*n) = []; 
        fprintf('[done]\n');
        
        % Add all the kinematic predictors
        Xfc = [Xfc, Xkin];
        
        % Fit GLMs for low and high uncertainty trials separately
        fprintf('\tFitting GLMs...');
        
        % % GLMFIT
        %[BLO, dev, stats] = glmfit(Xfc(TrainSELECT_LO,:), Yfc(TrainSELECT_LO,:), 'poisson');
        %[BHI, dev, stats] = glmfit(Xfc(TrainSELECT_HI,:), Yfc(TrainSELECT_HI,:), 'poisson');        
       
        % %-------GLMNET using R--------------------------------
        csvwrite('XforR.csv', Xfc(TrainSELECT_LO,:));
        csvwrite('YforR.csv', Yfc(TrainSELECT_LO,:));
        system('R CMD BATCH doGLMwithR.R ./Out.txt');
        BLO = csvread('BfromR.csv');
        
        csvwrite('XforR', Xfc(TrainSELECT_HI,:));
        csvwrite('YforR', Yfc(TrainSELECT_HI,:));
        system('R CMD BATCH doGLMwithR.R ./Out.txt');
        BHI = csvread('BfromR.csv');
        %-------------------------------------------------------

        % 
        %  % Elastic net
        %  [B, stats] = lassoglm(Xfc(TrainSELECT_LO,:), Yfc(TrainSELECT_LO,:), 'poisson', 'cv', 10, 'alpha', 0.1);
        %  BLO = B(:,stats.IndexMinDeviance);
        %  
        %  [B, stats] = lassoglm(Xfc(TrainSELECT_HI,:), Yfc(TrainSELECT_HI,:), 'poisson', 'cv', 10, 'alpha', 0.1);
        %  BHI = B(:,stats.IndexMinDeviance);
        
        compute_train_and_test_R2(Xfc, Yfc, TrainSELECT_LO, TestSELECT_LO, TrainSELECT_HI, TestSELECT_HI, BLO, BHI, L, n, cv);
        
        % [BLO, dev, stats] = glmfit(Xfc(find(LOvec == 1),:), Yfc(find(LOvec == 1),:), 'poisson');
        % [BHI, dev, stats] = glmfit(Xfc(find(HIvec == 1),:), Yfc(find(HIvec == 1),:), 'poisson');
    
        % % Fit a separate GLM with all data
        % [B, dev, stats] = glmfit(Xfc, Yfc, 'poisson');
    
        fprintf('[done]\n');
    
        fprintf('\tR2r_LO: %6.4f\t R2t_LO: %6.4f\n\t R2r_HI: %6.4f\t R2t_HI: %6.4f\n', R2r.LO.ALL(n,cv), R2t.LO.ALL(n,cv), R2r.HI.ALL(n,cv), R2t.HI.ALL(n,cv));
    
        % Collect the coefficients 
        ConMat_LO(n,cv,:) = BLO(2:end)';
        ConMat_HI(n,cv,:) = BHI(2:end)';
    end
end

%% Visualize fits (R2)

figure
for n=1:L    
    subplot(6,7,n); hold on;
    mu_tLO = [mean(R2t.LO.KIN(n,:)), mean(R2t.LO.SELF(n,:)), mean(R2t.LO.KINSELF(n,:)), mean(R2t.LO.SELFCROSS(n,:)), mean(R2t.LO.ALL(n,:))];
    si_tLO = [std(R2t.LO.KIN(n,:)), std(R2t.LO.SELF(n,:)), std(R2t.LO.KINSELF(n,:)), std(R2t.LO.SELFCROSS(n,:)), std(R2t.LO.ALL(n,:))];
    mu_tHI = [mean(R2t.HI.KIN(n,:)), mean(R2t.HI.SELF(n,:)), mean(R2t.HI.KINSELF(n,:)), mean(R2t.HI.SELFCROSS(n,:)), mean(R2t.HI.ALL(n,:))];
    si_tHI = [std(R2t.HI.KIN(n,:)), std(R2t.HI.SELF(n,:)), std(R2t.HI.KINSELF(n,:)), std(R2t.HI.SELFCROSS(n,:)), std(R2t.HI.ALL(n,:))];
    
    errorbar(1, mu_tLO(1), si_tLO(1), 'r', 'LineWidth', 2); plot(1, mu_tLO(1), 'ro', 'LineWidth', 2); 
    errorbar(2, mu_tHI(1), si_tHI(1), 'k', 'LineWidth', 2); plot(2, mu_tHI(1), 'ko', 'LineWidth', 2);
    
    errorbar(3, mu_tLO(2), si_tLO(2), 'r', 'LineWidth', 2); plot(3, mu_tLO(2), 'ro', 'LineWidth', 2); 
    errorbar(4, mu_tHI(2), si_tHI(2), 'k', 'LineWidth', 2); plot(4, mu_tHI(2), 'ko', 'LineWidth', 2);
    
    errorbar(5, mu_tLO(3), si_tLO(3), 'r', 'LineWidth', 2); plot(5, mu_tLO(3), 'ro', 'LineWidth', 2); 
    errorbar(6, mu_tHI(3), si_tHI(3), 'k', 'LineWidth', 2); plot(6, mu_tHI(3), 'ko', 'LineWidth', 2);
    
    errorbar(7, mu_tLO(4), si_tLO(4), 'r', 'LineWidth', 2); plot(7, mu_tLO(4), 'ro', 'LineWidth', 2); 
    errorbar(8, mu_tHI(4), si_tHI(4), 'k', 'LineWidth', 2); plot(8, mu_tHI(4), 'ko', 'LineWidth', 2);
    
    errorbar(9, mu_tLO(5), si_tLO(5), 'r', 'LineWidth', 2); plot(9, mu_tLO(5), 'ro', 'LineWidth', 2); 
    errorbar(10, mu_tHI(5), si_tHI(5), 'k', 'LineWidth', 2); plot(10, mu_tHI(5), 'ko', 'LineWidth', 2);
    
    plot([0:11], 0*[0:11], 'k--');
    axis([0 11 -0.15 0.25]);
    if(n <= L1)
        title(sprintf('M1 neuron: %d', NeuronSelect_M1(n)));
    else
        title(sprintf('PMd neuron: %d', NeuronSelect_PMd(n-L1)));
    end
end

%% Visualize kernels

%All -> All
figure
for n=1:L
    for i=1:L
        COLUMN = i; %if(i < n) COLUMN = i; else COLUMN = i+1; end
        %subplot(L,L,(n-1)*L+COLUMN); hold on;
        y1 = exp(Basis*squeeze(mean(ConMat_LO(n,:,3*(i-1)+1:3*i), 2)));
        y2 = exp(Basis*squeeze(mean(ConMat_HI(n,:,3*(i-1)+1:3*i), 2)));
        e1 = exp(Basis*squeeze(std(ConMat_LO(n,:,3*(i-1)+1:3*i), [], 2)))/sqrt(NCV);
        e2 = exp(Basis*squeeze(std(ConMat_HI(n,:,3*(i-1)+1:3*i), [], 2)))/sqrt(NCV);
        boundedline(iht, y1, e1, 'r', iht, y2, e2, 'k');
        xlabel(sprintf('%d',i)); ylabel(sprintf('%d',n))
        hold off; pause
        %plot(iht, exp(Basis*ConMat_LO(n,3*(i-1)+1:3*i)'), 'r', 'LineWidth', 2);
        %plot(iht, exp(Basis*ConMat_HI(n,3*(i-1)+1:3*i)'), 'k', 'LineWidth', 2);
        %ylim([0 1.5]);
        %axis('off'); box('off');
    end
end

%%
%M1 -> M1
figure
for n=1:L1
    for i=1:L1
        COLUMN = i; %if(i < n) COLUMN = i; else COLUMN = i+1; end
        subplot(L1,L1,(n-1)*L1+COLUMN); hold on;
        y1 = exp(Basis*squeeze(mean(ConMat_LO(n,:,3*(i-1)+1:3*i), 2)));
        y2 = exp(Basis*squeeze(mean(ConMat_HI(n,:,3*(i-1)+1:3*i), 2)));
        e1 = exp(Basis*squeeze(std(ConMat_LO(n,:,3*(i-1)+1:3*i), [], 2)))/sqrt(NCV);
        e2 = exp(Basis*squeeze(std(ConMat_HI(n,:,3*(i-1)+1:3*i), [], 2)))/sqrt(NCV);
        boundedline(iht, y1, e1, 'r', iht, y2, e2, 'k');
        
        %plot(iht, exp(Basis*ConMat_LO(n,3*(i-1)+1:3*i)'), 'r', 'LineWidth', 2);
        %plot(iht, exp(Basis*ConMat_HI(n,3*(i-1)+1:3*i)'), 'k', 'LineWidth', 2);
        %ylim([0 1.5]);
        %axis('off'); box('off');
    end
end

%%
%PMd -> PMd
figure
for n=L1+1:L
    for i=L1+1:L
        COLUMN = i-L1; %if(i < n) COLUMN = i-L1; else COLUMN = i-L1+1; end
        subplot(L2,L2,(n-L1-1)*L2+COLUMN); hold on;
        y1 = exp(Basis*squeeze(mean(ConMat_LO(n,:,3*(i-1)+1:3*i), 2)));
        y2 = exp(Basis*squeeze(mean(ConMat_HI(n,:,3*(i-1)+1:3*i), 2)));
        e1 = exp(Basis*squeeze(std(ConMat_LO(n,:,3*(i-1)+1:3*i), [], 2)))/sqrt(NCV);
        e2 = exp(Basis*squeeze(std(ConMat_HI(n,:,3*(i-1)+1:3*i), [], 2)))/sqrt(NCV);
        boundedline(iht, y1, e1, 'r', iht, y2, e2, 'k');
        
        %plot(iht, exp(Basis*ConMat_LO(n,3*(i-1)+1:3*i)'), 'r', 'LineWidth', 2);
        %plot(iht, exp(Basis*ConMat_HI(n,3*(i-1)+1:3*i)'), 'k', 'LineWidth', 2);
        %ylim([0 1.5]);
        %axis('off'); box('off');
    end
end

%%
%PMd -> M1
figure
for n=1:L1
    for i=L1+1:L
        COLUMN = i-L1;
        subplot(L1,L2,(n-1)*L2+COLUMN); hold on;
        y1 = exp(Basis*squeeze(mean(ConMat_LO(n,:,3*(i-1)+1:3*i), 2)));
        y2 = exp(Basis*squeeze(mean(ConMat_HI(n,:,3*(i-1)+1:3*i), 2)));
        e1 = exp(Basis*squeeze(std(ConMat_LO(n,:,3*(i-1)+1:3*i), [], 2)))/sqrt(NCV);
        e2 = exp(Basis*squeeze(std(ConMat_HI(n,:,3*(i-1)+1:3*i), [], 2)))/sqrt(NCV);
        boundedline(iht, y1, e1, 'r', iht, y2, e2, 'k');
        
        %plot(iht, exp(Basis*ConMat_LO(n,3*(i-2)+1:3*(i-1))'), 'r', 'LineWidth', 2);
        %plot(iht, exp(Basis*ConMat_HI(n,3*(i-2)+1:3*(i-1))'), 'k', 'LineWidth', 2);
        %ylim([0 1.5]);
        %axis('off'); box('off');
    end
end

%M1 -> PMd
figure
for n=1:L2
    for i=1:L1
        COLUMN = i;
        subplot(L2,L1,(n-1)*L1+COLUMN); hold on;
        y1 = exp(Basis*squeeze(mean(ConMat_LO(n+L1,:,3*(i-1)+1:3*i), 2)));
        y2 = exp(Basis*squeeze(mean(ConMat_HI(n+L1,:,3*(i-1)+1:3*i), 2)));
        e1 = exp(Basis*squeeze(std(ConMat_LO(n+L1,:,3*(i-1)+1:3*i), [], 2)))/sqrt(NCV);
        e2 = exp(Basis*squeeze(std(ConMat_HI(n+L1,:,3*(i-1)+1:3*i), [], 2)))/sqrt(NCV);
        boundedline(iht, y1, e1, 'r', iht, y2, e2, 'k');
        
        %plot(iht, exp(Basis*ConMat_LO(L1+n,3*(i-1)+1:3*i)'), 'r', 'LineWidth', 2);
        %plot(iht, exp(Basis*ConMat_HI(L1+n,3*(i-1)+1:3*i)'), 'k', 'LineWidth', 2);
        %ylim([0 1.5]);
        %axis('off'); box('off');
    end
end

%% Data-driven basis function learning

% Within PMd
s = 1;
CrossCorr = [];
for i=1:size(spikevec_PMd,2)
    fprintf('%d ', i);
    for j=i+1:size(spikevec_PMd,2)
        [CrossCorr(:,s), lags] = xcorr(log(eps+spikevec_PMd(:,i)), log(eps+spikevec_PMd(:,j)), 20, 'coeff');
        s = s+1;
    end
end

[~, PCs, E] = princomp(CrossCorr(21:end,:));
figure; plot(lags(21:end), PCs(:,1:5));