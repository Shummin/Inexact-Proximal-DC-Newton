function varargout = quad_BFGS_syyy(z, x0 , grad, s, y, sy, yy, ss)
% quadratic approximation of g with BFGS:
%   q(z) = 1/2 || z - x0||_{B_k} + grad'* ( z - x0 )
% B = I - ss'/ss + gamma*yy'/sy 
%   = N'*N, N = I + sqrt(gamma/(sy*ss))sy' - ss'/ss 
% H = I - (sy' + ys')/sy + (1/gamma + yy/sy)*ss'/sy
%   = M'*M, M = I + sqrt(1/(gamma*sy*ss))ss' - ys'/sy
% gamma = sy/yy;
% B = I - ss'/ss + yy'/yy, H = I - (sy' + ys')/sy + (2yy/sy)*ss^T/sy
% N = I + sqrt(1/(yy*ss))sy' - ss'/ss, M = I + sqrt(yy/ss)ss' - ys'/sy

zx = z - x0 ;
varargout{1} = zx + (y'*zx/yy)*y - (s'*zx/ss)*s + grad ; % gradient of q(z)
varargout{2} = z + (y'*z/yy)*y - (s'*z/ss)*s ;   % Bz

v = z + (sqrt(1/ss/yy)*(y'*z) - s'*z/ss)*s;
varargout{3} =  v'*v; % z'*B*z = ||Nz||^2
w = z + sqrt(yy/ss)*(s'*z)*s - s'*z/sy*y ;
varargout{4} = w'*w ; % z'*H*z = ||Mz||^2



end

