%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Example 2 : Least Squares with Log-Sum Penalty
% min 1/2 ||Ax-b||^2 + lambda*sum(log(1+abs(x)/epsilon))
% log(1+abs(x)/epsilon) = h1(x) - h2(x)
% h1(x) = 1/epsilon ||x||_1
% h2(x) = 1/epsilon ||x||_1 + sum(log(epsilon) - log(abs(x)+epsilon))
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all
clc

repeats = 20;
lambda_list = [1e-2 5e-3 1e-3 5e-4];
tol = 1e-5;

%% Begin loop for lambda and n
Table_ALL_mBFGS = []; Table_ALL_mSR1_V = [];
Table_ALL_pDCAe = [];   Table_ALL_nmAPG = [];
for h = 1: length(lambda_list)

    lambda = lambda_list(h);
    randn('seed',1);
    rand('seed',1);
    Table_AVE_mBFGS = []; Table_AVE_mSR1_V = [];
    Table_AVE_pDCAe = [];   Table_AVE_nmAPG = [];
    for l = 1:5
        m = 720*l;
        n = 2560*l;
        s = 80*l;
        
        epsilon = 0.5;
        h2 = @(x) logsum(x,lambda,epsilon);
        
        Table_mBFGS = []; Table_mBFGS_V= [];  Table_mSR1_V = [];
        Table_LBFGS = []; Table_pDCAe = [];   Table_nmAPG = [];
        
        for repeat = 1: repeats
            %% problem generation
            % B.Wen, X.Chen, T. K. Pong,
            % A proximal difference-of-convex algorithm with extrapolation
            % Computational Optimization and Applications, 2018
            A = randn(m,n);
            for ii = 1:n
                A(:,ii) = A(:,ii)/norm(A(:,ii));
            end
            err = 0.01*randn(m,1);
            I = randperm(n);
            J = I(1:s);
            y = zeros(n,1);
            y(J) = randn(s,1);
            b = A*y  + err;
            
            % compute Lipschitz constant L
            time_eig = 0;
            if m > 2000
                clear opts
                opts.issym = 1;
                tstart = tic;
                L = eigs(A*A',1,'LM',opts);
                time_eig = time_eig + toc(tstart);
            else
                tstart = tic;
                L = norm(A*A');
                time_eig = time_eig + toc(tstart);
            end
            fprintf('\n Lipschitz constant L = %g, time_eig = %g\n', L, time_eig)
            
            
            %% memoryless BFGS with semi-smooth Newton
            fprintf('******** memoryless BFGS (S-Newton) for Least squares and Log-Sum Penalty ********\n')
            clear opts;
            opts.x0 = zeros(n,1);
            opts.maxiter = inf; % max iteration
            opts.tol = tol; % tolerance
            opts.theta = 0.99; % inexactness
            opts.beta = 0.5;
            opts.delta = 0.5;
            opts.nu = 1.e-6;
            [x_mBFGS, iter_mBFGS, time_mBFGS, obj_mBFGS] = mlessBFGS_DC(A, b, lambda/epsilon, h2, opts);
            Table_mBFGS = [Table_mBFGS;...
                lambda m n iter_mBFGS time_mBFGS obj_mBFGS 0.5*norm(A*x_mBFGS-b)^2 nnz(x_mBFGS)];
             
            %% memoryless SR1 with V-FISTA
            fprintf('******** memoryless SR1 (V-FISTA) for Least squares and Log-Sum Penalty ********\n')
            clear opts;
            opts.x0 = zeros(n,1);
            opts.maxiter = inf; % max iteration
            opts.tol = tol; % tolerance
            opts.theta = 0.2; % inexactness
            opts.beta = 0.5;
            opts.delta = 0.5;
            opts.nu = 1.e-6;
            [x_mSR1_V, iter_mSR1_V, time_mSR1_V, obj_mSR1_V] = mlessSR1_DC_VFISTA(A, b, lambda/epsilon, h2, opts);
            Table_mSR1_V = [Table_mSR1_V;...
                lambda m n iter_mSR1_V time_mSR1_V obj_mSR1_V 0.5*norm(A*x_mSR1_V-b)^2 nnz(x_mSR1_V)];                  
            
            %% pDCAe
            fprintf('******** pDCAe for Least squares and Log-Sum Penalty ********\n')
            clear opts
            opts.x0 = zeros(n,1);
            opts.maxiter = inf;
            opts.tol = tol;
            [x_pDCAe, iter_pDCAe, time_pDCAe, obj_pDCAe] = ...
                pDCAe(A, b, lambda/epsilon, h2, L, opts);
            Table_pDCAe = [Table_pDCAe;...
                lambda m n iter_pDCAe time_pDCAe obj_pDCAe 0.5*norm(A*x_pDCAe-b)^2 nnz(x_pDCAe) time_eig];
            
            %% APG
            fprintf('******** nmAPG for Log-Sum Penalty ********\n')
            clear opts;
            regul = @(x,y) ProxLSP(x,y,lambda,epsilon,n);
            opts.x0 = zeros(n,1);
            opts.maxiter = inf; % max iteration
            opts.tol = tol; % tolerance
            [x_APG, iter_APG, time_APG, obj_APG] = APG(A, b, lambda, regul, opts);
            Table_nmAPG = [Table_nmAPG;...
                lambda m n iter_APG time_APG obj_APG 0.5*norm(A*x_APG-b)^2 nnz(x_APG)];
            
            fprintf('***                                                        ***\n')
            
            
        end
        fprintf('***                                                  ***\n')
        fprintf('mBFGS: (%4.e,%d,%d), iter = %4.1f, time = %5.2f, objval = %8.7f, 1/2||Ax-b|| = %8.7f, nnz(x) = %4.1f\n', ...
            mean(Table_mBFGS));
        fprintf('mSR1: (%4.e,%d,%d), iter = %4.1f, time = %5.2f, objval = %8.7f, 1/2||Ax-b|| = %8.7f, nnz(x) = %4.1f\n', ...
            mean(Table_mSR1_V));
        fprintf('pDCAe: (%4.e,%d,%d), iter = %4.1f, time = %5.2f, objval = %8.7f, 1/2||Ax-b|| = %8.7f, nnz(x) = %4.1f, time_eig = %5.2f\n', ...
            mean(Table_pDCAe));
        fprintf('nmAPG: (%4.e,%d,%d), iter = %4.1f, time = %5.2f, objval = %8.7f, 1/2||Ax-b|| = %8.7f, nnz(x) = %4.1f\n', ...
            mean(Table_nmAPG));
        Table_AVE_mBFGS   = [Table_AVE_mBFGS;mean(Table_mBFGS)]; 
        Table_AVE_mSR1_V  = [Table_AVE_mSR1_V;mean(Table_mSR1_V)]; 
        Table_AVE_pDCAe   = [Table_AVE_pDCAe;mean(Table_pDCAe)]; 
        Table_AVE_nmAPG   = [Table_AVE_nmAPG;mean(Table_nmAPG)];
    end
	Table_ALL_mBFGS   = [Table_ALL_mBFGS;Table_AVE_mBFGS]; 
	Table_ALL_mSR1_V  = [Table_ALL_mSR1_V;Table_AVE_mSR1_V]; 
	Table_ALL_pDCAe   = [Table_ALL_pDCAe;Table_AVE_pDCAe]; 
	Table_ALL_nmAPG   = [Table_ALL_nmAPG;Table_AVE_nmAPG];
end

% export
T1 = array2table(Table_ALL_mBFGS,'VariableNames',...
    {'lambda','sample','dim','iter','time','objval','SSR','nnz'});
writetable(T1,'LSP_mBFGS.csv')
T2 = array2table(Table_ALL_mSR1_V,'VariableNames',...
    {'lambda','sample','dim','iter','time','objval','SSR','nnz'});
writetable(T2,'LSP_mSR1_V.csv')
T3 = array2table(Table_ALL_pDCAe,'VariableNames',...
    {'lambda','sample','dim','iter','time','objval','SSR','nnz','time_eig'});
writetable(T3,'LSP_pDCAe.csv')
T4 = array2table(Table_ALL_nmAPG,'VariableNames',...
    {'lambda','sample','dim','iter','time','objval','SSR','nnz'});
writetable(T4,'LSP_nmAPG.csv')



