function [x_new, iter, time, objval] = LBFGS_DC(A, b,lambda, DC_h2, opts)
%   solves the following optimization by the proposed method with L-BFGS + tfocs:
%       minimize f(x) = g(x) + (h1(x) - h2(x)) =g(x)  + lambda*(||x||_1 - ||x||)
%  this method can be regarded as a DCA version of PNOPT (https://web.stanford.edu/group/SOL/software/pnopt/)

tic()
% Parameters
[~, n] = size(A);
if isfield(opts,'x0'),       x0 = opts.x0;              else x0 = zeros(n,1);end
if isfield(opts,'maxiter'),  maxiter = opts.maxiter;    else maxiter = inf;  end
if isfield(opts,'tol'),      tol = opts.tol;            else tol = 1e-6;     end
if isfield(opts,'theta'),    theta = opts.theta;        else theta = 0.9;    end
if isfield(opts,'beta'),     beta = opts.beta;          else beta = 0.5;     end
if isfield(opts,'delta'),    delta = opts.delta;        else delta = 0.5;    end
if isfield(opts,'nu'),       nu = opts.nu;              else nu = 1.e-6;     end
if isfield(opts,'Lbfgs_mem'),LBFGS_mem = opts.LBFGS_mem;else LBFGS_mem = 10; end


% initialization
iter = 1;
x = x0 ;
x_old = x0 ;
N = length(size(x0)) ;
y = zeros(N,1);
s = zeros(N,1);
grad_old = zeros(N,1);
sPrev = zeros( length(x), 0 );
yPrev = zeros( length(x), 0 );

% l1 penalty, tfocs's option
l1_pen  = prox_l1(lambda);
tfocs_opts = struct(...
    'alg'        , 'N83' ,...
    'printEvery' , 0     ...
    );

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

%%  proximal quasi-Newton
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
    
    % prox, limited memory BFGS (L-BFGS) 
    % prox = argmin 0.5*||x-(x_k-H(\nabla g(x_k)-\xi_k)||^2 + lambda|x|_1 
    if size( sPrev, 2 ) > LBFGS_mem
        sPrev = [ sPrev(:,2:LBFGS_mem), s ];
        yPrev = [ yPrev(:,2:LBFGS_mem), y ];
        scale = yy/sy;
    else
        sPrev = [ sPrev, s ]; 
        yPrev = [ yPrev, y ]; 
        scale = yy/sy;
    end
    B_x = LBFGS_Prod( sPrev, yPrev, scale );
    quad_sub = @(z) quad_LBFGS( B_x, grad2, gx, h2, z - x );
    tfocs_opts.tol =  max( 0.1*tol, 0.01 * optim ) ;
    x  = tfocs( quad_sub, [], l1_pen, x, tfocs_opts );
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

fprintf('L-BFGS (tfocs): iter = %d, time = %5.2f, objval = %8.7f, nnz(x) = %d\n', iter, time, objval, nnz(x_new));


end

  
  
function B_x = LBFGS_Prod( S, Y, D ) 
% L-BFGS Hessian approximation
  l = size( S, 2 );
  L = zeros( l );
  for k = 1:l
    L(k+1:l,k) = S(:,k+1:l)' * Y(:,k);
  end
  d1 = sum( S .* Y );
  d2 = sqrt( d1 );
  
  R    = chol( D * ( S' * S ) + L * ( diag( 1 ./ d1 ) * L' ), 'lower' );
  R1   = [ diag( d2 ), zeros(l); - L*diag( 1 ./ d2 ), R ];
  R2   = [- diag( d2 ), diag( 1 ./ d2 ) * L'; zeros( l ), R' ];
  YdS  = [ Y, D * S ];
  B_x  = @(x) D * x - YdS * ( R2 \ ( R1 \ ( YdS' * x ) ) );
end

function [quad, grad_q] = quad_LBFGS( B, grad, g, h2, d )
  grad_q = B(d) + grad;
  quad = d' * ( 0.5*B(d) + grad  ) + g - h2;
  
end

