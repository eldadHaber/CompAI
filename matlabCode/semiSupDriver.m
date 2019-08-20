% driver for semisupervised learning
clear 
close all
X = randn(2,10000);


R  = sqrt(X(1,:).^2 + X(2,:).^2);
i1 = R < 0.2;
i2 = (0.4 < R) & (R < 0.6);
i3 = (0.8 < R) & (R < 1);

X1 = X(:,i1);
X2 = X(:,i2);
X3 = X(:,i3);

X = [X1, X2, X3];

Xn =  2*rand(2,400)-1;



n1 = size(X1,2); 
n2 = size(X2,2);
n3 = size(X3,2);

X = [X1, X2, X3, Xn];

plot(X(1,:),X(2,:),'.k','MarkerSize',25)

nd = 2;
I = [1:nd,n1+[1:nd],n1+n2+[1:nd]];

hold on
plot(X(1,I(1:nd)),X(2,I(1:nd)),'.r','markersize',30);
plot(X(1,I(nd+1:2*nd)),X(2,I(nd+1:2*nd)),'.b','markersize',30);
plot(X(1,I(2*nd+1:3*nd)),X(2,I(2*nd+1:3*nd)),'.g','markersize',30);
plot(X(1,I(1:2:nd)),X(2,I(1:2:nd)),'.r','markersize',30);
plot(X(1,I(nd+1:2:2*nd)),X(2,I(nd+1:2:2*nd)),'.b','markersize',30);
hold off

% setup the labels with hot one
C = zeros(3,nd*3);
C(1,1:nd) = 1;
C(2,nd+1:2*nd) = 1;
C(3,2*nd+1:3*nd) = 1;


% Compute the appropriate GL
nn      = 9;
[A,dd] = getAdjacencyMatrix(X,nn);
figure
spy(A)
epsilon = median(dd(:));
[L,W,D] = getGraphLaplacian(X,A,epsilon);
figure(3)
spy(L) 

% choose some parameters
param.alpha   = 0.1;
param.beta    = 1e-3*norm(L,'inf');
param.mu      = 1e-1;
param.maxIter = 200;
% Call the function

[U,Cp] = semiSuperLearn(X,I,C,L,param);

figure(3)
[vals,ii] = max(Cp,[],1); 
i1 = ii==1;
i2 = ii==2;
i3 = ii==3;

X1 = X(:,i1);
X2 = X(:,i2);
X3 = X(:,i3);

figure(4)
plot(X1(1,:),X1(2,:),'.r',X2(1,:),X2(2,:),'.b',X3(1,:),X3(2,:),'.g')
    