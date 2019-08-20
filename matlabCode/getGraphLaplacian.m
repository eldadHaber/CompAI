function [L,W,D] = getGraphLaplacian(X,A,epsilon,isnormal)

if nargin < 4
    isnormal=1;
end
n = size(A,1);
[I,J,~] = find(A);

V = exp(-sum((X(:,I)-X(:,J)).^2,1)/epsilon);
W = sparse(I,J,V,n,n);
D = diag(sparse(sum(W,2)));
L = D-W;

if isnormal>0
    Dh = diag(sparse(1./sqrt(diag(D))));
    L = Dh*L*Dh; 
end

% get rid of small a-symmetries
L = 0.5*(L+L');

