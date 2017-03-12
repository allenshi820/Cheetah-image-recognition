function[p_err] = classification(mask, I, mean_FG, pi_FG, var_FG, mean_BG, pi_BG, var_BG, dim, C)

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
        A = A';
        X = A(1:dim,:);
        
        %%classify  block A with the # = dim features
        state_FG = 0;
        state_BG = 0;
        for class = 1:C
            var_FG(:,:,class) = var_FG(:,:,class) .* eye(dim, dim);
            var_BG(:,:,class) = var_BG(:,:,class) .* eye(dim, dim);
            state_FG = state_FG + log(mvnpdf(X',mean_FG(:,class)',var_FG(:,:,class))) + log(pi_FG(class));
            state_BG = state_BG + log(mvnpdf(X',mean_BG(:,class)',var_BG(:,:,class))) + log(pi_BG(class));
        end
        
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
    image(:,263:270) = 0;
    image(248:255,:) = 0;
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

    
    
    
end



