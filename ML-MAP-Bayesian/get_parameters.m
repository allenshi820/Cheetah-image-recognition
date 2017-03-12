function[mean_FG, covariance_FG, mean_BG, covariance_BG] = get_parameters(alpha_number, dataset)
data = importdata('TrainingSamplesDCT_subsets_8.mat');
%data1 = importdata('Prior_1.mat');
data1 = importdata('Prior_2.mat');
alpha = importdata('Alpha.mat');
if dataset ==1
    FG = data.D1_FG;
    BG = data.D1_BG;
end
if dataset ==2
    FG = data.D2_FG;
    BG = data.D2_BG;
end
if dataset ==3
    FG = data.D3_FG;
    BG = data.D3_BG;
end
if dataset ==4
    FG = data.D4_FG;
    BG = data.D4_BG;
end
%calculate the covariance 
cov_BG = cov(BG);
cov_FG = cov(FG);
%calculate u0 and cov0 in the pdf of parameter mu
mu_FG = data1.mu0_FG;
mu_FG = mu_FG';
mu_BG = data1.mu0_BG;
mu_BG = mu_BG';
cov_mu = zeros(64,64);
alpha_value = alpha(1,alpha_number);
w = data1.W0;
for i=1:64
    cov_mu(i,i) = alpha_value*w(1,i);
end

%calculate the posterior mean and covariance for the distribution of mu
%given class training data
%mean and cov for BG
sample_mu_BG = mean(BG)';
[size_BG, ~] = size(BG);
n = 1/size_BG;
mu_BG_n = cov_mu * ((cov_mu + n*cov_BG)^-1) * sample_mu_BG + n * cov_BG * ((cov_mu + n*cov_BG)^-1) * mu_BG;
cov_BG_n = cov_mu * ((cov_mu + n*cov_BG)^-1) * n*cov_BG;

%mean and cov for FG
sample_mu_FG = mean(FG)';
[size_FG, ~] = size(FG);
n = 1/size_FG;
mu_FG_n = cov_mu * ((cov_mu + n*cov_FG)^-1) * sample_mu_FG + n * cov_FG * ((cov_mu + n*cov_FG)^-1) * mu_FG;
cov_FG_n = cov_mu * ((cov_mu + n*cov_FG)^-1) * n*cov_FG;


%computer the parameters of the predictive distribution for BG
mean_BG = mu_BG_n;
covariance_BG = cov_BG + cov_BG_n;

%computer the parameters of the predictive distribution for BG
mean_FG = mu_FG_n;
covariance_FG = cov_FG + cov_FG_n;
end

