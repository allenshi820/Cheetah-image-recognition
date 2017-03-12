clc;clear;
I = imread('cheetah.bmp');
I = im2double(I);
TrainsampleDCT_BG = importdata('TrainsampleDCT_BG.mat');
TrainsampleDCT_FG = importdata('TrainsampleDCT_FG.mat');


FG = 0.1918; %Prior probability of forground
BG = 0.8081; %Prior probability of background
count = 1;

image = zeros(64714,1);

%calculate the covariance matrices for the best 8 features
cov_FG = cov(TrainsampleDCT_FG(:,[1,11,20,23,34,35,39,40]));
cov_BG = cov(TrainsampleDCT_BG(:,[1,11,20,23,34,35,39,40]));
%calculate the mean for the best 8 features
mean_BG = mean(TrainsampleDCT_BG(:,[1,11,20,23,34,35,39,40]));
mean_FG = mean(TrainsampleDCT_FG(:,[1,11,20,23,34,35,39,40]));
mean_BG = mean_BG';
mean_FG = mean_FG';

for i = 0:246
    
    for j = 0:261
        %start

        %A = I(i+1:i+8,j+1:j+8);
        m = i+1;
        A = zeros(8,8);
        x = 1;
        
        for a = 1:8  %assign the values in 8x8 blocks starting in the first row
            n = j+1;  %get the current columncoun
            y = 1;    %reset to the first rown in A
            for b = 1:8 %column
                A(x,y) = I(m,n);     %assign the values in 8x8 blocks to A
                y = y + 1;
                n = n + 1;
            end
            x = x + 1;      %update row number in 8x8 block
            m = m + 1;      %update row number in A
            
        end
       

        
        
        %end
        A = dct2(A);
        %A is zig-zag pattern, reshape it
        ind = reshape(1:numel(A), size(A));         %# indices of elements
        ind = fliplr( spdiags( fliplr(ind) ) );     %# get the anti-diagonals
        ind(:,1:2:end) = flipud( ind(:,1:2:end) );  %# reverse order of odd columns
        ind(ind==0) = [];
        A = A(ind);
        A = A';
        %X = A;
        X = A([1,11,20,23,34,35,39,40],:);
        
        %%classify the8x8 block A with the best 8 features
        state_FG = log(mvnpdf(X,mean_FG,cov_FG)) + log(FG);
        state_BG = log(mvnpdf(X,mean_BG,cov_BG)) + log(BG);
        if state_FG > state_BG
            image(count,1) = 1;
        else
            image(count,1) = 0;
        end
    
        count = count+1;
        
           
    end
    
end
image = reshape(image, [262,247]);
image = image';
a = zeros(247,8);
image = [image a];
a = zeros(8,270);
image = [image; a];
imagesc(image);
colormap(gray(255));

%---------------------------calculate the probability of error---------------------------
mask = imread('cheetah_mask.bmp');
mask = im2double(mask);
%get the sum of 1's
count_one = sum(sum(mask==1));
%get one_diff
true = (mask==1);
false = (image==1);
same = true&false;
one_diff = sum(sum(true-same));
%get the sum of 0's
count_zero = sum(sum(mask==0));
%get zero_diff
true = (mask==0);
false = (image==0);
same = true&false;
zero_diff = sum(sum(true-same));
p_err = (one_diff / count_one)*0.2 + (zero_diff / count_zero)*0.8;

