function [h2,subgrad] = ell2(x,lambda)
% h2 : lambda||x||_2
% subgrad : subgradient of lambda||x||_2

    tmp = norm(x);
    h2 = lambda*tmp; 
    if tmp > 0
        subgrad = lambda*x/tmp;
    else
        subgrad = lambda*x;
    end
end

