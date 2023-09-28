function xt = ProxB_VFISTA( xk, quadsub, theta, tol, lambda,L, mu )
%  solves the following subproblem:
%       minimize f(x) = 1/2||x-(x_k-H_k\nabla g(x_k)||^2_B_k + lambda*||x||_1,

% initialization
t = 0; 
max_iter = 10000 ;
xt = xk ;
Nes = sqrt(L/mu) ;
xt_old = xt ;

% first step
gradq = quadsub(xt_old) ;
xt = soft_thresh( xt_old - gradq/L , lambda/L) ;
d = xt - xk;
Nes = (Nes-1)/(1+Nes) ;

%%  MAIN LOOP
for t = 1:max_iter
  t =  t + 1 ;
  
  % Check for termination
  [~,~, dBd, ~] = quadsub(d);
  r1 = xt_old - xt ; 
  [~, r] = quadsub(r1) ;
  r = L*r1 - r ;
  [~, ~, ~,r] = quadsub(r);
  if  r <= theta*dBd || norm(d) < tol || norm(r1) <= 1e-12
    break ;
  end
  
  % Update sequance
  xt = xt  + Nes* (xt-xt_old);
  xt_old = xt ;
  gradq = quadsub(xt) ;
  xt = soft_thresh( xt_old - gradq./L , lambda/L) ;
  d = xt - xk;
  

end

end
