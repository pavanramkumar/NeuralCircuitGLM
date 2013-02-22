
function [X,S] = getX(S,bas,const,padZeros,verbose)

if nargin<3, const=0; end
if nargin<4, padZeros=1; end
if nargin<5, verbose=0; end

M = size(bas,1);
X = repmat(0,size(S,1)*M,size(S,2));

if verbose, tic; end
c = (1:M:size(S,1)*M) - 1;
for m=1:M
    X(c+m,:) = filter(bas(m,:),1,full(S),[],2);
%     X(c+m,:) = filtfilt(bas(m,:),1,full(S)')';
    if verbose, fprintf('.'); end
end

if verbose, fprintf('\n'); toc; end

if ~padZeros
    X = X(:,size(bas,2):end);
    if nargout>1, S = S(:,size(bas,2):end); end
end
if const, X = [repmat(1,1,size(X,2)); X]; end