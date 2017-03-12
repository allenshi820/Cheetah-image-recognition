%This function performs the Bayesian estimation for different data set and
%different alpha values in both strategies

function[p_err] = bayes_classification(dataset)
mask = imread('cheetah_mask.bmp');
mask = im2double(mask);
alpha = importdata('Alpha.mat');
p_err = zeros(9,1);
%classification
for alpha_number=1:9
    [mean_FG, covariance_FG, mean_BG, covariance_BG] = get_parameters(alpha_number, dataset);
    I = imread('cheetah.bmp');
    I = im2double(I);
    p_FG = 0.2;
    p_BG = 0.8;


    count = 1;
    image = zeros(64714,1);
    for i = 0:246
    
        for j = 0:261
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
            
            A = dct2(A);
            %A is zig-zag pattern, reshape it
            ind = reshape(1:numel(A), size(A));         %# indices of elements
            ind = fliplr( spdiags( fliplr(ind) ) );     %# get the anti-diagonals
            ind(:,1:2:end) = flipud( ind(:,1:2:end) );  %# reverse order of odd columns
            ind(ind==0) = [];
            A = A(ind);
            X = A';
        
        
            state_FG = log(mvnpdf(X,mean_FG,covariance_FG)) + log(p_FG);
            state_BG = log(mvnpdf(X,mean_BG,covariance_BG)) + log(p_BG);
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
    %imagesc(image);
    %colormap(gray(255));
    
    
    %calculate the probability of error
    count_one = 0;
    count_zero = 0;
    one_diff = 0;
    zero_diff = 0;
    for x = 1:255
        for y = 1:270
            if mask(x,y) == 1
                count_one = count_one + 1;
                if image(x,y) == 0
                    one_diff = one_diff + 1;
                end
            end   
            if mask(x,y) == 0
                count_zero = count_zero + 1;   
                if image(x,y) == 1 
                    zero_diff = zero_diff + 1;
                end     
            end    
        end
    end

p_one_error = one_diff / count_one;
p_zero_error = zero_diff / count_zero;
p_err(alpha_number,1) = p_one_error * p_FG + p_zero_error * p_BG;
end

semilogx(alpha,p_err,'b');
xlabel('alpha value');
ylabel('probability of error');
end