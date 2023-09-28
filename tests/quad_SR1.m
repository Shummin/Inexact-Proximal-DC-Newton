function varargout = quad_SR1(z, x0 , grad, w, w2, deno, deno2)
% quadratic approximation of g with SR1:
%   q(z) = 1/2 || z - x0||_{B_k} + grad'* ( z - x0 )
% SR1: B = I + w*w'/deno, H = I + w2*w2'/deno2

  zx = z - x0 ;
  varargout{1} = zx + w'*zx/deno*w + grad ; % gradient of q(z)
  varargout{2} = z + w'*z/deno*w ;          % Bz
  varargout{3} = z'*(z + w'*z/deno*w);      % z'*B*z
  varargout{4} = z'*(z + w2'*z/deno2*w2);   % z'*H*z

end

