function [x_new, iter, time, objval] = pDCAe(A, b, lambda, DC_h2, L, opts)
%%%%%%%%%%%%%%%%%%%% pDCAe %%%%%%%%%%%%%%%%%%%%%%%%%%
% Wen, B., Chen, X., Pong, T.K.:
% A proximal difference-of-convex algorithm with extrapolation.
% Computational Optimization and Applications 69(2), 297?324 (2018)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% solves the following optimization by pDCAe:
% min 0.5*||Ax - b||^2 + lambda*||x||_1 - DC_h2(x),
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tic();

% Parameters
[~, n] = size(A);
if isfield(opts,'x0'),       x0 = opts.x0;           else x0 = zeros(n,1);end
if isfield(opts,'maxiter'),  maxiter = opts.maxiter; else maxiter = inf;  end
if isfield(opts,'tol'),      tol = opts.tol;         else tol = 1e-6;     end

% Initialization
x_old = x0;
x_new = x_old;
Ax_old = A*x_old;
Ax_new = Ax_old;
theta = 1;
theta0 = 1;
iter = 1;
h1 = lambda*norm(x_new,1) ;
h2 = DC_h2(x_new) ;
gx = 1/2*norm(Ax_new - b)^2;
objval = gx + h1 - h2;

%% main loop
while iter < maxiter 
    % Subgradient of DC_h2(x)
    [~,tmp] = DC_h2(x_new);
    
    % Extrapolation
    beta = theta0/theta - 1/theta;
    u = x_new + beta*(x_new - x_old);
    Au = Ax_new+ beta*(Ax_new - Ax_old);
    
    % Proximal gradient
    u_tmp = u - (1/L)*(A'*(Au - b) - tmp);
    x_old = x_new;
    Ax_old = Ax_new;
    x_new = soft_thresh(u_tmp,lambda/L);
    Ax_new = A*x_new;
    
    % update objval
    h1 = lambda*norm(x_new,1) ;
    h2 = DC_h2(x_new) ;
    gx = 1/2*norm(Ax_new - b)^2;
    objval = gx + h1 - h2;

    % Check for termination
    if  norm(x_new - x_old)/max(1,norm(x_new)) <= tol
        iter = iter + 1;
        break
    end
    
    % Update theta
    theta0 = theta;
    theta = (1 + sqrt(1+4*theta^2))/2;
    
    if mod(iter,200) == 0
        theta0 = 1;
        theta = 1;
    end
    
    iter = iter + 1;
end
time = toc();

fprintf('pDCAe: iter = %d, time = %5.2f, objval = %8.7f, nnz(x) = %d\n', iter, time, objval, nnz(x_new));
end
