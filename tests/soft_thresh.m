function s = soft_thresh(c,lambda)
% solves the following problem:
%    min 0.5*||x-c||^2 + lambda*||x||_1
% s : soft thresholding
s = sign(c).*max(0,abs(c) - lambda);
end