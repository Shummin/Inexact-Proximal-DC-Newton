function [x_new, iter, time, objval] = mlessSR1_DC_VFISTA(A, b,lambda, DC_h2, opts)
%  solves the following optimization by the proposed method with melssSR1 + V-FISTA:
%       minimize f(x) = g(x) + (h1(x) - h2(x)) =g(x)  + lambda*(||x||_1 - ||x||)

tic()
% Parameters
[~, n] = size(A);
if isfield(opts,'x0'),       x0 = opts.x0;           else x0 = zeros(n,1);end
if isfield(opts,'maxiter'),  maxiter = opts.maxiter; else maxiter = inf;  end
if isfield(opts,'tol'),      tol = opts.tol;         else tol = 1e-6;     end
if isfield(opts,'theta'),    theta = opts.theta;     else theta = 0.9;    end
if isfield(opts,'beta'),     beta = opts.beta;       else beta = 0.5;     end
if isfield(opts,'delta'),    delta = opts.delta;     else delta = 0.5;    end
if isfield(opts,'nu'),       nu = opts.nu;           else nu = 1.e-6;     end

% initialization
iter = 1;
x = x0 ;
x_old = x0 ;
% quasi-Newton
N = length(size(x0)) ;
y = zeros(N,1);
s = zeros(N,1);
grad_old = zeros(N,1);
theta = (1.-theta)*(1.-theta) ;

%%               MAIN LOOP
% first step (proximal gradient, B0 = I) %
Ax = A*x;
Axb= Ax - b;
grad = A'*(Axb) ;
gx = 1/2*norm(Axb)^2;
h1 = lambda*norm(x,1) ;
[h2, tmp] = DC_h2(x);
f11 = gx + h1 - h2 ;

x = soft_thresh(x - (grad-tmp),lambda);
Ax = A*x;
Axb = A'*(Ax-b);
gx = 1/2*norm(Axb)^2;
d = x - x_old ;
eta = 1 ;    % stepsize
h1_old = h1;
h1 = lambda*norm(x,1) ;
h2 = DC_h2(x) ;
line_lam = delta*((grad-tmp)'*d + h1 - h1_old) ;
fx = gx + h1 - h2;
while fx > f11  + eta*line_lam
    eta = beta*eta ;
    x = x_old + eta*d ;
    Ax = A*x;
    gx = 1/2*norm(Ax - b)^2;
    h1 = lambda*norm(x,1) ;
    h2 = DC_h2(x) ;
    fx = gx + h1 - h2;
end

% Check for termination
optim = norm( d , 'inf' ) / max(1, norm( x, 'inf' ))  ;
if  optim <= tol
    objval = fx;
    disp('solution at 1st step');
    x_new = x; time = toc() ;
    return
end

%%  memoryless proximal quasi-Newton
iter = 2;
while iter < maxiter
    
    % gradient A'(Ax-b), subgradient of h2, s, y
    s = x - x_old ;
    x_old = x ;
    grad_old = grad ;
    Ax = A*x;
    Axb= Ax - b;
    grad = A'*(Axb) ;
    y = grad - grad_old ;
    [~, tmp] = DC_h2(x);
    grad2 = grad - tmp;
    ss = s'*s ; sy = s'*y ;
    if sy <= nu*ss
        y = y + (max(0,-sy/ss) + nu)*s; sy = dot(s,y); 
    end 
    yy = y'*y ;
    
    % prox, SR1 
    % prox = argmin 0.5*||x-(x_k-H(\nabla g(x_k)-\xi_k)||^2 + lambda|x|_1 
    gamma =  0.8*sy/yy ;
    w_H  = s - gamma*y ;
    w_B  = gamma*y - s;
    denoH = gamma*y'*w_H ;
    denoB = w_B'*s ;
    quad_sub = @(z)  quad_SR1(z, x_old ,grad2, w_B, w_H, denoB, denoH);
    x = ProxB_VFISTA( x_old,  quad_sub, theta, tol, lambda, 1 , 1 + w_B'*w_B/denoB  );
    d = x - x_old ;
  
    Ax = A*x;
    Axb= Ax - b;
    gx = 1/2*norm(Axb)^2;
    h1_old = h1;
    h1 = lambda*norm(x,1) ;
    h2 = DC_h2(x);
    objval = gx + h1 - h2;
    % Check for termination
    optim = norm( d ) / max( 1, norm(x) ) ;
    if   optim <= tol 
        x_new = x;
        break ;
    end
    
    % line search
    line_lam = grad2'*d + h1 - h1_old ;
    line = 0;
    line_lam2 = delta*line_lam ;
    f11= fx;
    fx = objval;
    eta = 1;
    while fx - f11 >  eta*line_lam2
        eta = eta*beta ;
        x = x_old + eta*d ;
        Ax = A*x;
        gx = 1/2*norm(Ax - b)^2;
        h1 = lambda*norm(x,1) ;
        h2 = DC_h2(x);
        fx = gx + h1 - h2;
        line = line + 1;
        if line == 50
            break ;
        end
    end
    objval = fx;
    
    x_new = x;
    iter = iter + 1;
    
end
%%
time = toc() ;

fprintf('Memoryless SR1 (V-FISTA): iter = %d, time = %5.2f, objval = %8.7f, nnz(x) = %d\n', iter, time, objval, nnz(x_new));




end

