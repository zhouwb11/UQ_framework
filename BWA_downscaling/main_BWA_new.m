clear;close all;clc;
% This example is to show how to run MCMC process in the Bayesian Weighted
% Averaging approach of Tebaldi C et al. 2004.

% add path for where MCMC.m 
path(path,'/Users/username/Desktop/BWA_downscaling');

% load data
G_obs = load('G_obs.mat'); % historical ground heat flux observation, N1 x 1
G_cs = load('G_cs.mat');  % control period, i.e., historical ground heat flux output from GCMs, N1 x M
G_ft_ssp245 = load('G_ft_ssp245.mat'); % ssp245 scenario predicted ground heat flux output from GCMs, N1 x M
G_ft_ssp585 = load('G_ft_ssp585.mat'); % ssp585 scenario predicted ground heat flux output from GCMs, N1 x M

% N1 and N2 are the number of days in the historical and future periods
% M is the number of GCMs. In our study, N1 = 12418 (1981-2014),
% N2 = 31411 (2015-2100), and M = 9. Different GCMs may have different
% number of days in a year, so a preprocessing step needs to unify N1 and N2 across GCMs.


% select Scenario
Scen = 1; % 1, ssp245; 2, ssp585

% Number of GCM used
NMOD = size(G_cs,2);

% MCMC parameters
rng(0);
N_montecarlo = 75000; 
save_every   = 50;
burn_in      = 25000;
STDTX = 1; % Natural variability for data: Standard deviation
Nmc          = 1000;  % The number of samples for posterior in Monte-Carlo simulations
RDM1=unifrnd(0,1,1,Nmc); % random numbers generated from the continuous uniform distributions



if Scen == 1
    G_ft = G_ft_ssp245;
elseif Scen == 2
    G_ft = G_ft_ssp585;
end

% compute annual mean G fluxes
G_cs = mean(G_cs,1); % 1981-2014

% split the future period into three segments and run BWA for each separately
G_ft = mean(G_ft(1:9497,:),1); % 2015-2040
% G_ft = mean(G_ft(9498:20454,:),1); % 2041-2070
% G_ft = mean(G_ft(20455:end,:),1); % 2071-2100

G_obs = mean(G_obs);


% run MCMC
[mu_G_BWA,ni_G_BWA,theta,lam_G_BWA,beta] = MCMC(G_cs(1,:),G_ft(1,:),G_obs(1),STDTX/G_obs(1),N_montecarlo,save_every,burn_in);
%mu_Ta_BWA: posterior of weighted G flux for the control period
%ni_Ta_BWA: posterior of weighted G flux for the future period
%DT_fut_M: posterior of weighted G flux changes between future and control periods
%%%
snm=sort(ni_G_BWA-mu_G_BWA);
cnm=(1:length(snm))/(length(snm)+1);  % [snm cnm]: marginal CDF
for kgb=1:Nmc
    DT_fut_M(kgb)=interp1(cnm,snm,RDM1(kgb)); %(y,x,yi)
end

% Plot for the BWA posterior
[f1,xi1] = ksdensity(mu_G_BWA);
[f2,xi2] = ksdensity(ni_G_BWA);
[f3,xi3] = ksdensity(DT_fut_M);

% mean G flux, control
mean_G_CT = mean(mu_G_BWA)
% median G flux, control
median_G_MCT = median(mu_G_BWA)


% mean G flux, future
mean_G_FT = mean(ni_G_BWA)
% median G flux, future
median_G_MFT = median(ni_G_BWA)




figure(1)
scatter(G_cs(M,:),zeros(NMOD,1),72,'kd'); hold on; grid on;
for i = 1:9
    if i == 1
        Model = 'ACCESS';
    elseif i == 2
        Model = 'CESM';
    elseif i == 3
        Model = 'EC_Earth3';
    elseif i == 4
        Model = 'HadGEM';
    elseif i == 5
        Model = 'IPSL';
    elseif i == 6
        Model = 'MIROC';
    elseif i == 7
        Model = 'MPI';
    elseif i == 8
        Model = 'NorESM';
    elseif i == 9
        Model = 'UKESM';
    end
    text(G_cs(1,i),0.1,num2str(i),'FontSize',15);
end
scatter(G_obs(1),0,72,'ro','filled'); hold on; grid on;
plot(xi1,f1,'b-','LineWidth',4);
plot([median_G_MCT median_G_MCT],[0 max(f1)],'k--','LineWidth',3)
xlabel('$\mathrm{\overline{G}\ historical, [W/m^2]}$','Interpreter','Latex');
ylabel('$\mathrm{pdf}$','Interpreter','Latex');
set(gca,'FontSize',20);
leg = legend('GCMs','Obs','BWA');
leg.FontSize = 15; leg.FontWeight = 'bold';
box on;

figure(2)
scatter(G_ft(1,:),zeros(NMOD,1),72,'kd'); hold on; grid on;
for i = 1:9
    if i == 1
        Model = 'ACCESS';
    elseif i == 2
        Model = 'CESM';
    elseif i == 3
        Model = 'EC_Earth3';
    elseif i == 4
        Model = 'HadGEM';
    elseif i == 5
        Model = 'IPSL';
    elseif i == 6
        Model = 'MIROC';
    elseif i == 7
        Model = 'MPI';
    elseif i == 8
        Model = 'NorESM';
    elseif i == 9
        Model = 'UKESM';
    end
    text(G_ft(1,i),0.1,num2str(i),'FontSize',15);
end
plot(xi2,f2,'b-','LineWidth',4);
plot([median_G_MFT median_G_MFT],[0 max(f2)],'k--','LineWidth',3)
xlabel('$\mathrm{\overline{G}\ future, [W/m^2]}$','Interpreter','Latex');
ylabel('$\mathrm{pdf}$','Interpreter','Latex');
set(gca,'FontSize',20);
box on;

