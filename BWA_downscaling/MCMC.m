%-------------------------------------------------------------------------%
% PARAMETER OF MULTIMODEL ENSEMBLE (MARKOV CHAIN MONTE CARLO)
% Reference: Tebaldi et al 2004; 2005; Tebaldi and Knutti 2007
%-------------------------------------------------------------------------%
function[mu_s,ni_s,theta_s,lambda_s,beta_s] = MCMC(X,Y,X0,CVX,N_montecarlo,save_every,burn_in)
%-------------------------------------------------------------------------%
% INPUT
% X [1,NMOD]: Models outputs of control scenario
% Y [1,NMOD]: Models outputs of future scenario
% X0[1]     : Observations
% OUTPUT
% mu = control scenario mean
% ni = future scenario mean
% lambda1 = [1/var] model weigths
% theta [parameter for precision in reproducing future climate]
%-------------------------------------------------------------------------%
ANSW_st = 0;     % Introducing heavy-tails 
ANSW_b  = 1;     % Correlation coefficient between future and present climate 
ANSW_theta = 1;  % Deflation inflation parameter 
FI=4;
X=reshape(X,1,length(X)); 
Y=reshape(Y,1,length(X)); 
%%%%%%%%%%%%%%%%%%%
lambda0=1/((CVX*X0).^2); %%% [1/var] uncertainty observations
%%%%%%%%%%%%%%%%%%%%%%%%
NMOD = length(X);
r=0;
a=0.01; b=0.01;
%%% INITIAL VALUES
lambda= 1*ones(1,NMOD);
mu = mean(X); mu_tilde=mu;
ni = mean(Y); ni_tilde=ni;
theta = 1;
beta = 0; beta_tilde=beta;
for j = 1:N_montecarlo
    %%%%%%% MARKOV CHAIN MONTE CARLO
    if ANSW_st == 1
        %%%%%%%% si
        A = 0.5*(FI+1);
        B = 0.5*(FI + lambda.*((X-mu).^2));
        for i=1:NMOD
            s(i) = gamrnd(A,1/B(i));
        end
        %%%%%%%% ti
        A = 0.5*(FI+1);
        B = 0.5*(FI + theta*lambda.*((Y-ni-beta*(X-mu)).^2));
        for i=1:NMOD
            t(i) = gamrnd(A,1/B(i));
        end
    else
        t=ones(1,NMOD);
        s=ones(1,NMOD);
    end
    %%%%%%%%%%%%% Theta
    if ANSW_theta == 1
        A = a+ NMOD/2;
        B = b + 0.5*(sum(t.*lambda.*(Y-ni-beta.*(X-mu)).^2));
        theta = gamrnd(A,1/B);
    else
        theta=1;
    end
    %%%%%%%%%%%%% Lambda
    A = a+1;
    B = b+ 0.5*s.*(X-mu).^2 + 0.5*theta*t.*((Y-ni-beta.*(X-mu)).^2);
    for i=1:NMOD
        lambda(i) = gamrnd(A,1/B(i));
    end
    %%%%%%%%%%%%% Beta
    if ANSW_b == 1
        beta_tilde = sum(t.*lambda.*(Y-ni).*(X-mu))/sum(t.*lambda.*((X-mu).^2));
        M = beta_tilde;
        S = (theta*sum(t.*lambda.*(X-mu).^2)).^-1;
        beta = normrnd(M,sqrt(S));
    else
        beta=0;
    end
    %%%%%%%%%%%%%%% Mu
    mu_tilde = (sum(s.*lambda.*X) - theta.*beta*sum(t.*lambda.*(Y-ni-beta*X)) + lambda0*X0)/(sum(s.*lambda) + ...
        theta*(beta^2)*sum(t.*lambda) +lambda0);
    M = mu_tilde;
    S = (sum(s.*lambda) + theta*(beta^2)*sum(t.*lambda) + lambda0)^-1;
    mu = normrnd(M,sqrt(S));
    %%%%%%%%%%%%%%%%% Ni
    ni_tilde = sum(t.*lambda.*(Y-beta*(X-mu)))/sum(t.*lambda);
    M = ni_tilde;
    S = (theta*sum(t.*lambda))^-1;
    ni = normrnd(M,sqrt(S)); 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%% burn_in=25000; save_every=50;
    if (j>burn_in) && (mod(j,save_every)==0)
        r = r+1;
        mu_s(r) = mu;
        ni_s(r) = ni;
        theta_s(r) = theta;
        lambda_s(r,:) = lambda;
        beta_s(r,:) = beta;
    end
end
end