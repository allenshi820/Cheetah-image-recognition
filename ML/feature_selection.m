clc;clear;

TrainsampleDCT_BG = importdata('TrainsampleDCT_BG.mat');
TrainsampleDCT_FG = importdata('TrainsampleDCT_FG.mat');
mean_BG = zeros(64,1);
mean_FG = zeros(64,1);
var_BG = zeros(64,1);
var_FG = zeros(64,1);

%calculate background covariance
%%cov_FG = cov(TrainsampleDCT_FG(:,:));

for i = 1:64
    %calculate the mean
    mean_BG(i,1) = mean(TrainsampleDCT_BG(:,i));
    mean_FG(i,1) = mean(TrainsampleDCT_FG(:,i));
    
    %calculate the variance
    var_BG(i,1) = var(TrainsampleDCT_BG(:,i));
    var_FG(i,1) = var(TrainsampleDCT_FG(:,i));

end



features = [3,4,59,60,61,62,63,64];
count = 1;
for j=1:8
        i = features(1,j);
        x_BG = (mean_BG(i,1)- 3 * sqrt(var_BG(i,1))):0.0001:(mean_BG(i,1)+ 3 * sqrt(var_BG(i,1)));
        x_FG = (mean_FG(i,1)- 3 * sqrt(var_FG(i,1))):0.0001:(mean_FG(i,1)+ 3 * sqrt(var_FG(i,1)));
        
        norm_BG = normpdf(x_BG, mean_BG(i,1), sqrt(var_BG(i,1)));
        norm_FG = normpdf(x_FG, mean_FG(i,1), sqrt(var_FG(i,1)));
        subplot(3,3,count);
        plot(x_BG, norm_BG, x_FG, norm_FG);
        title(['feature'  num2str(i)]);
        count = count + 1;
end  

    
