%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%   function spiketrains = simulateNetwork(baseline, K, L, beta, X)
%   This script simulates a spiking network with
% - external covariates, X: L x M,
%   where M is number of covariates,
%   L is length of spiketrain in ms
% - weights to combine external covariates, beta: M x 1
% - self and cross term kernels, K: N x N x p,
% - baseline firing rates of individual neurons, baseline: N x 1
% - N is number of neurons, p is temporal support of kernels
% 
% Refer Emery Brown et al. (2001) Neural Computation
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function spiketrains = simulateNetwork(baseline, K, L, beta, X)

%%%%% Test parameters %%%%%
% K = randn(4,4,20);
% L = 1000;
% beta = [];
%%%%%%%%%%%%%%%%%%%%%%%%%%%

if(nargin < 3)
    beta = [];
end


N = size(K,1);
p = size(K,3);

% Initialize spiketrain with at least p zero spikes
spiketrains = zeros(N, p);
TAU = exprnd(1,N,1);
%Proceed ms by ms

U = zeros(N,1);

for t=p+1:L+p
    % Generate the probability of spiking for each neuron at time 't'
    spike = zeros(N,1);
    for n1=1:N
        h = zeros(N,p);
        % Accumulate the contribution of each neuron to neuron n1
        for n2=1:N
            h(n1,:) = h(n1,:) + filter(squeeze(K(n1,n2,:)), 1, spiketrains(n2, t-p:t-1)); 
        end
        
        % Generate expected ISI
        if(isempty(beta))
            % Intrinsic connectivity only
            U(n1) = U(n1) + exp(baseline(n1) + h(n1,p));
        else
            % Intrinsic connectivity and extrinsic covariates
            U(n1) = U(n1) + exp(baseline(n1) + h(n1,p) + X(t,:)*beta);
        end
    end
    
%     fprintf('t: %d ms\n', t); 
%     fprintf('\t U: ');
%     for n=1:N
%         fprintf('%6.4f ', U(n));
%     end
%     fprintf('\n');
%     
%     fprintf('\t TAU: '); 
%     for n=1:N
%         fprintf('%6.4f ', TAU(n));
%     end
%     fprintf('\n');
    
    for n1=1:N
        % Apply time-rescaling principle to generate spike
        if(U(n1) > TAU(n1))
            spike(n1,1) = 1;
            TAU(n1) = exprnd(1);
            U(n1) = 0;
        end
    end
    spiketrains = [spiketrains, spike];
    
%     for n=1:N
%         fprintf('\t %d ', spike(n,1));
%     end
%     fprintf('\n');
end



