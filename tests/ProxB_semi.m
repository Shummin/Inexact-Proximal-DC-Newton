function xt = ProxB_semi( xk, lambda, quadsub, theta, d0, U1, U2)
%  solves the following subproblem:
%   Prox_h^B(v) = argmin h(x) + 1/2 || x - v ||^2_B
%   B = I + U1U1^T - U2U2^T

% initialization
alp = [0;0] ;
PU1 = U1/(1+U1'*U1); % (I+U1U1^T)^{-1} = I - U1 PU1^T
P1U2 = U2-(PU1'*U2)*U1; % (I+U1U1^T)^{-1}U2
xx = xk + d0 ;
t = 1;
% first step
zeta_alp = xx - U1*alp(1)  + P1U2*alp(2);
prox = soft_thresh( zeta_alp,lambda);
L = [U1'*(xx - prox + P1U2*alp(2))+alp(1);
    U2'*(xx - prox)+alp(2)];
LL_new = L'*L/2 ;

%%  MAIN LOOP
while(t <= 50)
    d = prox - xk ;
    
    % Check for termination
    r = -L(1)*U1 + L(2)*U2;
    [~, ~, dBd, ~] = quadsub(d);
    [~, ~, ~,rHr] = quadsub(r);
    if rHr <= theta*dBd %&& norm(L) < 1e-6
        break;
    end
    
    % semismooth Newton step
    p1 = U1; p2 = P1U2;
    idx1 = find( abs(zeta_alp) > lambda  );
    J = [(U1(idx1)'*p1(idx1)+1) (U1'*P1U2-U1(idx1)'*p2(idx1));
        U2(idx1)'*p1(idx1) 1-(U2(idx1)'*p2(idx1))];
    invJ = [J(2,2) -J(1,2);-J(2,1) J(1,1)]/( J(1,1)*J(2,2) - J(1,2)*J(2,1) );
    d_L = -invJ*L;
    LL_old = LL_new ;
    alp_old = alp ;
    alp = alp_old + d_L ;
    zeta_alp = xx - U1*alp(1) + P1U2*alp(2);
    prox = soft_thresh( zeta_alp,lambda);
    L = [U1'*(xx - prox + P1U2*alp(2))+alp(1);
        U2'*(xx - prox)+alp(2)];
    
    % line search
    LL_new = L'*L/2;
    beta = 1;
    for l = 1:10
        if LL_new <= LL_old - 0.0002*beta*LL_old
            break;
        end
        beta = beta/2 ;
        alp = alp_old + beta*d_L ;
        zeta_alp = xx - U1*alp(1) + P1U2*alp(2);
        prox = soft_thresh( zeta_alp,lambda);
        L = [U1'*(xx - prox + P1U2*alp(2))+alp(1);
            U2'*(xx - prox)+alp(2)];
        LL_new = L'*L/2 ;
    end
    t = t+1;
end
xt = prox;

end

