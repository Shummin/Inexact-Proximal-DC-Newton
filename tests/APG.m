function [x_new, iter, time, objval] = APG( A, b, lambda, regul, opts)
%%%%%%%%%%%%%% nonmonotone APG %%%%%%%%%%%%%%%%%%%%%%
% Li, H., Lin, Z.:
% Accelerated proximal gradient methods for nonconvex programming. In:
% Advances in neural information processing systems, pp. 379?387 (2015)
% (Algorithm 4 in Supplementary)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% solves the following optimization by nonmonotone APG:
% min 0.5*||Ax - b||^2 + regul(x),
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tic()
% Parameters
[m, n] = size(A);
if isfield(opts,'x0'),       x0 = opts.x0;           else x0 = zeros(n,1); end
if isfield(opts,'maxiter'),  maxiter = opts.maxiter; else maxiter = inf;  end
if isfield(opts,'tol'),      tol = opts.tol;         else tol = 1e-6;      end

% initialization
x_old = x0;
x_new = x_old;
z = x_old;
x_oldold = x_old;
iter = 1;
y = x_new;
Ax2 = A*x_new ;
grady = A'*(Ax2 - b);
ssr = 0.5*norm(Ax2-b)^2;
h = regul(x_new,lambda) ;
objval = ssr + h;
delta = 0.0001/2;
c = objval;
q = 1;
theta0 = 0;
theta  = 1;

%% main loop
while iter < maxiter
    old_y = y;
    y = x_new + (theta0/theta)*(z-x_new) + (theta0-1)/theta*(x_new-x_oldold);
    
    % proximal gradient
    Ax = A*y ;
    grad_oldy = grady;
    grady = A'*(Ax - b) ;
    if iter == 1
        eta = 1;
    else
        eta = min(1.e+8,max(norm(y-old_y)^2/((y-old_y)'*(grady-grad_oldy)),1.e-8));
    end
    [~,x_new] = regul(y - eta*grady,lambda*eta);
    z = x_new;
    
    Ax2 = A*x_new ;
    ssr = 0.5*norm(Ax2-b)^2;
    
    h = regul(x_new,lambda) ;
    hy = regul(y,lambda) ;
    objval = ssr + h ;
    Fy = 0.5*norm(Ax-b)^2 + hy;
    while objval > Fy - delta*norm(y-z)^2 && objval > c - delta*norm(z-y)^2
        eta = 0.5*eta;
        [~,x_new] = regul(y - eta*grady,lambda*eta);
        z = x_new;
        
        Ax2 = A*x_new ;
        ssr = 0.5*norm(Ax2-b)^2;
        h = regul(x_new,lambda) ;
        objval = ssr + h;
    end
    
    if objval > c - delta*norm(z-y)^2
        Ax = A*x_old ;
        grad = A'*(Ax - b) ;
        if iter == 1
            eta = 1;
        else
            eta = min(1.e+8,max(norm(x_old-old_y)^2/((x_old-old_y)'*(grad-grad_oldy)),1.e-8));
%             eta = min(1.e+8,max(((x_old-old_y)'*(grad-grad_oldy))/norm(grad-grad_oldy)^2,1.e-8));
        end
        [~,v] = regul(x_old - eta*grad,lambda*eta);
        
        Ax2 = A*v ;
        ssr = 0.5*norm(Ax2-b)^2;
        h = regul(v,lambda) ;
        objval2 = ssr + h;
        while objval2 > c - delta*norm(x_old-v)^2
            eta = 0.5*eta;
            [~,v] = regul(x_old - eta*grad, lambda*eta);
            Ax2 = A*v ;
            ssr = 0.5*norm(Ax2-b)^2;
            h = regul(v,lambda) ;
            objval2 = ssr + h;
        end
        if objval >  objval2
            x_new = v;
            objval = objval2;
        end
    end

    theta0 = theta;
    theta = (sqrt(4*theta*theta + 1)+1)/2;
    q_old = q;
    q = 0.2*q+1;
    c = (0.2*q_old*c + objval)/q;
    
    % Check for termination
    if norm(x_new-x_old)/max(1,norm(x_new)) <= tol
        iter = iter + 1;
        break ;
    end
    x_oldold = x_old;
    x_old = x_new;
    
    iter = iter + 1;
    
end
time = toc();

fprintf('nmAPG: iter = %d, time = %5.2f, objval = %8.7f, nnz(x) = %d\n', iter, time, objval, nnz(x_new - x_old));


end









