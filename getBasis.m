
function bas = getBasis(typ,n,m,varargin)

bas = [];
switch lower(typ)
    case 'hist'
        bas = getBasis_his(n,m,varargin);
    case 'log'
        bas = getBasis_log(n,m,varargin);
    case 'exp'
        bas = getBasis_exp(n,m,varargin);
    case 'rcos'
        bas = getBasis_rcos(n,m,varargin);
    case 'rand'
        bas = getBasis_rand(n,m,varargin);
    case 'gamma'
        bas = getBasis_gamma(n,m,varargin);
    otherwise
        disp(['ERR: unknown basis ' typ '...']); return;
end
bas = [zeros(size(bas,1),1) bas]; % padding to make filters work
% bas = bas./repmat(sum(bas,2),1,m+1);

function bas = getBasis_his(n,m,varargin)
    if m<n, disp('ERR: need m>n to make linear filters...'); return; end

    bas = zeros(n,m);
    bs = floor(m/n);
    for i=1:n
        bas(i,((i-1)*bs+1):(i*bs)) = 1;
    end
	bas(end,n*bs:end)=1;
    
function bas = getBasis_log(n,m,varargin)
    if m<n, disp('ERR: need m>n to make linear filters...'); return; end
    
    ind = unique(floor(logspace(0,log10(m),n+1)));
    bas = zeros(min(n,length(ind)-1),m);
    for i=1:length(ind)-1
        bas(i,ind(i):ind(i+1)-1) = 1;
    end
    bas(end,ind(end):end)=1;
    
function bas = getBasis_exp(n,m,varargin)
    tmax = m;
    tau = logspace(log10(1/tmax),log10(tmax),n);
%     tau = linspace((1/tmax),(tmax/2),n);
    
    bas = repmat(1:m,n,1);
    for i=1:n
        bas(i,:) = tau(i)*exp(-tau(i)*bas(i,:));
    end
    
function bas = getBasis_rcos(n,m,varargin)
    st = 1; clip = 1;
    if ~isempty(varargin{1})
        st = varargin{1}{1};
        if length(varargin{1})>1, clip = varargin{1}{2}; end
    end
        
    [t, ihbas, bas] = makeRaisedCosBasis(n, 1, [0 m/2], st);

%     keyboard
    bas = bas';
    if clip
        if size(bas,1)>m
            bas = bas(:,1:m,:);
        else
            bas = [bas zeros(n,m-size(bas,2)-1)];
        end
    end
    
function bas = getBasis_rand(n,m,varargin)
    bas = randn(n,m);
    
function bas = getBasis_gamma(n,m,varargin)
    if ~isempty(varargin{1})
        s = varargin{1}{1};
    else
        s = 1;
    end
    t = linspace(0,10000,10000);
    tmp = t.^(n^s).*exp(-t/m);
    tmp = tmp/max(tmp);
    [null, imax] = max(tmp);
    tmax = min(find(tmp(imax:end)<10e-3))+imax;
    tmp = tmp(1:tmax);
    bas = zeros(n,tmax);
    t = linspace(0,tmax,tmax);
    for i=0:(n-1)
        bas(i+1,:) = t.^(i^s).*exp(-t/m);
        bas(i+1,:) = bas(i+1,:)/max(bas(i+1,:));
    end