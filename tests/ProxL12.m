function [h, x] = ProxL12(y,lambda)
% l1 - l2
% h : lambda(||y||_1 - ||y||_2)
% x = argmin_x .5||x-y||^2 + lambda(||x||_1 - ||x||_2)

h = lambda*(norm(y,1)-norm(y));

x = zeros(size(y));

if max(abs(y)) > 0 
    if max(abs(y)) > lambda
        x   = max(abs(y) - lambda, 0).*sign(y);
        x   = x * (norm(x) + lambda)/norm(x);
    else
        if max(abs(y)) > 0
            [~, i]  = max(abs(y));
            x(i(1)) = abs(y(i(1)))* sign(y(i(1)));
        end
    end
end
end