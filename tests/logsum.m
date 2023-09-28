function [h2,subgrad] = logsum(x,lambda,epsilon)
% h2 : lambda *?\sum_i (|x_i|/epsilon - log(epsilon + |x_i|)) + log(epsilon))
% subgrad : subgradient of h2

    h2 = lambda*(sum(abs(x)/epsilon + log(epsilon)) - sum(log(epsilon+abs(x))));     
    subgrad = lambda * x./(epsilon*(epsilon + abs(x))) ;
end

