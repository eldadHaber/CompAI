% simple driver for semisupervised learning
clear 
close all

nsamples = 100;

X = [randn(2,nsamples/2), randn(2,nsamples/2)+4]; 
X = X(:,randperm(nsamples));

figure(1)
plot(X(1,:),X(2,:),'.r','MarkerSize',50);

nn      = 3;
[A,dd] = getAdjacencyMatrix(X,nn);
epsilon = median(dd(:))*100;
L = getGraphLaplacian(X,A,epsilon,0);
figure(2)
spy(L) 

% Now compute the eigenvalues
[V,D] = eig(full(L));

%for i=1:10
figure(3)
i = 2;
    scatter(X(1,:),X(2,:),1500,V(:,i),'.');
    %pause;
%end
return
% choose some parameters
    