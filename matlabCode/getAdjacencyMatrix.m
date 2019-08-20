function [A,dd] = getAdjacencyMatrix(data,nn)
%  A = CreateAdjacency(data,nn)
% 

n = size(data,2);
I = zeros((nn+1),n); 
J = zeros((nn+1),n);

dd = zeros(n,nn+1);
for i=1:n
    r = data - data(:,i);    
    d = sum(r.*r,1);

    [di,idx] = sort(d);
    I(:,i) = i*ones(nn+1,1);
    J(:,i) = idx(1:nn+1);    
    dd(i,:) = di(1:nn+1)';
end


A = sparse(I(:),J(:),ones(length(J(:)),1),n,n);
A = sign(A+A');
    