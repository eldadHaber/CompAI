function[U,Cpred] = semiSuperLearn(X,I,C,L,param)
% [C] = semiSuperLearn(X,I,C,L,param)
% Compute the label of the (unlabeled) data X given
% the a few labeled indices ind
% Solve the problem
%
% min 0.5*u'*L(X)*u + alpha*SM(u(:,I),C) 
%

nl = length(I);  % number of labeled points
nx = size(X,2);    % number of examples
nc = size(C,1);    % number of classes

% generate projection matrix
P = sparse(1:nl,I,ones(nl,1),nl,nx);

% Initialize the solution
U = zeros(nc,nx);

% start optimization
iter = 1;
muls = param.mu;
while 1
    [F,dF] = softmaxLoss(U*P',C);
    dF     = dF*P;

    % Compute the objective
    f  = 0.5*param.alpha*trace(U*L*U') + F;
    df = param.alpha*L*U' + dF';

    % Compute a Newton like step
    dU = (param.alpha*(L + param.beta*speye(size(L,1))))\df;
    lsIter = 1;
    while 1
        Utry = U - muls*dU';
        Ftry = softmaxLoss(Utry*P',C);
        ftry = 0.5*param.alpha*trace(Utry*L*Utry') + Ftry;
        
        if ftry < f
            break;
        end
        muls = muls/2;
        lsIter = lsIter+1;
        if lsIter>8
            disp('LSB');
        end
    end
    if lsIter == 1
        muls = muls*1.3;
    end
    U = Utry;
    Cpred = exp(U)./sum(exp(U),1);
    % make some pictures
    figure(2)
    [~,ii] = max(Cpred,[],1); 
    i1 = ii==1;
    i2 = ii==2;
    i3 = ii==3;
    
    X1 = X(:,i1);
    X2 = X(:,i2);
    X3 = X(:,i3);


    %plot(X1(1,:),X1(2,:),'.r',X2(1,:),X2(2,:),'.b',X3(1,:),X3(2,:),'.g')
%     scatter(X(1,:),X(2,:),20,Cpred(1,:) + 10*Cpred(2,:) +  100*Cpred(3,:));
    scatter(X(1,:),X(2,:),20,Cpred','.');
    drawnow;
    
    fprintf('%3d   %3.2e    %3.2e    %3.2e    %3.2e\n',iter,f,norm(df(:)),norm(dU(:)),muls)
    iter = iter+1;
    if iter>param.maxIter
        return
    end
end
    