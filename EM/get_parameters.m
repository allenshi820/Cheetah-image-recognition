function[mean_FG, pi_FG, var_FG, mean_BG, pi_BG, var_BG] = get_parameters(FG, BG, random, random1, iter, dim, C)

%--------------------- parameters for FG and BG---------------------------%
%initialize mean, variance and pi randomly for all C classes
mean_FG = FG(random:random+C-1,1:dim)';
mean_BG = BG(random1:random1+C-1,1:dim)';
for i = 1:C
    var_FG((i-1)*dim+1:i*dim,1:dim) = cov(FG(:,1:dim));
    var_BG((i-1)*dim+1:i*dim,1:dim) = var_FG((i-1)*dim+1:i*dim,1:dim);
end
pi_FG(1:C)= 1/C;
pi_BG(1:C)= 1/C;
x_FG = FG(:,1:dim);
x_BG = BG(:,1:dim);


%---------------------------- EM Algorithm -------------------------------%
for i = 1:iter
    
    %---------------------------- FG -------------------------------------%
    
    mean_prev_FG = mean_FG;
    mean_prev_BG = mean_BG;
    
    for j = 1:C
        mean_curr = mean_FG(:,j);
        var_curr = var_FG((j-1)*dim+1:j*dim,1:dim);
        pi_curr = pi_FG(j);
        
        %calculate the sum of hij and mean_numer
        sum_hij_deno = 0;
        for m = 1:C
            sum_hij_deno = sum_hij_deno + mvnpdf(x_FG,mean_FG(:,m)', var_FG((m-1)*dim+1:m*dim,1:dim))*pi_FG(m);
        end
        numer = mvnpdf(x_FG, mean_curr', var_curr)*pi_FG(j);
        sum_hij = sum(numer./sum_hij_deno);
        
        numer = repmat(numer, [1,dim]);
        sum_hij_deno = repmat(sum_hij_deno, [1,dim]);
        
        mean_numer = sum((numer.*x_FG)./sum_hij_deno);
        mean_temp = (mean_numer / sum_hij)';
        
              
        var_numer = 0;
        for n = 1:250
            constant = 0;
            for m = 1:C
                constant = constant + mvnpdf(x_FG(n,:), mean_FG(:,m)', var_FG((m-1)*dim+1:m*dim,1:dim))*pi_FG(m);
            end
            var_numer = var_numer + ((mvnpdf(x_FG(n,:), mean_curr', var_curr).* pi_curr.* ((x_FG(n,:)' - mean_temp)*(x_FG(n,:)' - mean_temp)')) ./ constant);
        end
        
        %update mean
        mean_FG(:,j) = mean_temp;
        %make sure the covariance is always postive definite
        val = (var_numer / sum_hij);
        if det(val) < 1e-4
            val = val + eye(dim,dim)*(0.05);
        end
        %update covariance
        var_FG((j-1)*dim+1:j*dim,:) = val;

    
        pi_FG(j) = (1/250)*sum_hij;
        
   
        if (i ~= 1) && (norm(mean_prev_FG(1,:) - mean_FG(1,:)) < 1e-2)
            break;
        end
        
    end
    
   

%-------------------------------- BG -------------------------------------%
    
    for j = 1:C
        mean_curr = mean_BG(:,j);
        var_curr = var_BG((j-1)*dim+1:j*dim,1:dim);
        pi_curr = pi_BG(j);
        
        %calculate the sum of hij
        sum_hij_deno = 0;
        for m = 1:C
            sum_hij_deno = sum_hij_deno + mvnpdf(x_BG,mean_BG(:,m)', var_BG((m-1)*dim+1:m*dim,1:dim))*pi_BG(m);
        end
        numer = mvnpdf(x_BG, mean_curr', var_curr)*pi_BG(j);
        sum_hij = sum(numer./sum_hij_deno);
        
        numer = repmat(numer, [1,dim]);
        sum_hij_deno = repmat(sum_hij_deno, [1,dim]);
        
        mean_numer = sum((numer.*x_BG)./sum_hij_deno);
        mean_temp = (mean_numer / sum_hij)';
        
        %calculate the sum of hij*(xi - u)^2 - variance numerator
        var_numer = 0;
        for n = 1:1053
            constant = 0;
            for m = 1:C
                constant = constant + mvnpdf(x_BG(n,:), mean_BG(:,m)', var_BG((m-1)*dim+1:m*dim,1:dim))*pi_BG(m);
            end
            var_numer = var_numer + ((mvnpdf(x_BG(n,:), mean_curr', var_curr).* pi_curr .* ((x_BG(n,:)' - mean_temp)*(x_BG(n,:)' - mean_temp)')) ./ constant);
        end
        
        %update mean
        mean_BG(:,j) = mean_temp;
        %make sure the covariance is always postive definite
        val1 = (var_numer / sum_hij);
        if det(val1) < 1e-4
            val1 = val1 + eye(dim,dim)*(0.05);
        end
        
        %update covariance
        var_BG((j-1)*dim+1:j*dim,:) = val1;

        
        %update pi
        pi_BG(j) = (1/1053)*sum_hij;
    end
        
 
    %break when the sum 
    if (norm(mean_prev_BG(1,:) - mean_BG(1,:)) < 1e-2) && (i ~= 1)
        break;
    end
end

    



end