function[p_err, image] = classification(para)

mask = im2double(imread('cheetah_mask.bmp'));
I = im2double(imread('cheetah.bmp'));
count = 1;
image = zeros(64714,1);
for i = 0:246
    
    for j = 0:261
        
        A = I(i+1:i+8,j+1:j+8);
        A = dct2(A);
        %A is zig-zag pattern, reshape it
        ind = reshape(1:numel(A), size(A));         %# indices of elements
        ind = fliplr( spdiags( fliplr(ind) ) );     %# get the anti-diagonals
        ind(:,1:2:end) = flipud( ind(:,1:2:end) );  %# reverse order of odd columns
        ind(ind==0) = [];
        A = A(ind);
        X = A';

        state = para(1,2:65) * X + para(1,1);
        
        if state > 0
            image(count,1) = 1;
        end
        if state < 0
            image(count,1) = 0;
        end
     
    
        count = count+1;
        
           
    end
    
end

    image = reshape(image, [262,247]);
    image = image';
    image(:,263:270) = 0;
    image(248:255,:) = 0;
    image = double(image);
    imshow(image);
    figure;
    imagesc(image);
    colormap(gray(255));
    
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


