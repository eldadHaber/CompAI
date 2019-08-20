function [F,dF] = softmaxLoss(U,C)
% Cobs = exp(U)./(e'*exp(U))
% F = C(:)'*log(Cobs(:))

nex  = size(C,2);
            
S    = exp(U);          
F    = -sum(sum(C.*(U))) + sum(log(sum(S,1)));
F    = F/nex;

dF   = (-C + S./sum(S,1))/nex; %S./sum(S,2));
