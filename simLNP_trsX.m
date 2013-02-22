function firings = simLNP_trsX(alph,delta,lim,beta,X)

C = size(alph,1);
M = (size(alph,2)-1)/C;
if ~isfinite(M), M=0; end

if nargin<4, beta=[]; end

maxrows = ceil(lim/delta);

% Preallocate for speed
firings = zeros(maxrows,2);
erands = exprnd(1,maxrows,1);

fcount = zeros(C,1);
I = [1 zeros(1,C*M)]';
t = 1;

u = zeros(C,1); % clocks
tau = exprnd(1,C,1); % next spike time

ind = (1:C)';
ttp = ones(C,1);
smax =1;

cont = true;

% keyboard

tic
while cont
    cont = (t*delta)<lim;

    if isempty(beta)
        u = u + delta*exp(alph*I);
    else
        u = u + delta*exp(alph*I + beta*X(t,:)');
    end
    
    fired = u>tau;
             
    if sum(fired)
        if (smax+sum(fired) > size(firings,1))
            firings = [firings; zeros(maxrows,2)];  % allocate some more space
        end
        firings(smax:(smax+sum(fired)-1),:) = [t*ttp(1:sum(fired)) ind(fired)];
        fcount = fcount+fired;
        smax = sum(fcount);

        if (smax+sum(fired) > length(erands))
            erands = [erands; exprnd(1,maxrows,1)];  % allocate some more space
        end
%         tau(fired) = exprnd(1,sum(fired),1);
        tau(fired) = erands(smax:(smax+sum(fired)-1));
        u(fired) = 0;
    end
    
    %     I = circshift(I',1)';
    I(2:end) = I(1:end-1);
    I(1) = 1;
	if M>0,
        I(2:M:C*M+1) = fired;
    end
    
    if (mod(t,100000) == 0)
        fprintf('t:%04i >> %04i spikes (%04i min)\n',t*delta,round(mean(fcount)),min(fcount));
        % Sometings wrong...
        if min(fcount) == 0
            cont = false;
        end
    end
    
    t = t+1;
end

firings = firings(1:sum(fcount)-1,:);
firings(:,1) = firings(:,1)*delta;

fprintf('t:%04i >> %04i spikes\n',t*delta,min(fcount));
toc