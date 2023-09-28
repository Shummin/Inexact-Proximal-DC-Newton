function [val, prox] = ProxLSP(y,lam_eta,lambda,epsilon,n)
% Log Sum Penalty:
% val = lambda*sum(log(1+abs(x)/epsilon))
% prox = argmin lam_eta*sum(log(1+abs(x)/epsilon)) + 1/2 ||x-y||^2

% loss value
val = lambda*sum(log(1+abs(y)./epsilon)); 

%% comput proximal operater
% initialization
prox = zeros(n,1); 
absy = abs(y);

% 
delta = (epsilon - absy).^2 - 4*(lam_eta - epsilon*absy);
ind = (delta >= 0);

if isempty(ind) == 0
    % 
    ytmp = absy(ind);
    t1 = max((ytmp - epsilon + sqrt(delta(ind)))/2,0);
    
    % compare function values
    fun = 0.5*(t1 - ytmp).^2 + lam_eta*log(1+ t1/epsilon) - 0.5*ytmp.^2;
    ind1 = (fun < 0);
    
    % set proximal operater
    ind(ind) = ind1;
    prox(ind) = t1(ind1);
    prox = sign(y).*prox;
end

end

