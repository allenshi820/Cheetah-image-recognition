function[w1, w2] = MLP(eta1, eta2, max_step)

data = importdata('TrainingSamplesDCT_8_new.mat');
FG = data.TrainsampleDCT_FG;
BG = data.TrainsampleDCT_BG;
%training_set = 1303x65
training_set(:,1:64) = [FG;BG];
training_set(:,65) = 1;
a1 = training_set;
y = zeros(1303,1);
y(1:250,1) = 1;
y(251:1303,1) = 0;
y_enc = zeros(2,1303);
y_enc(1,1:250) = 1;
y_enc(1,251:13-3) = 0;
y_enc(2,251:1303) = 1;
y_enc(2,1:250) = 0;

%to initialize the weights w1 and w2
w1 = rand(30,65);       %w1 = 30x65
w2 = rand(2,31);        %w2 = 2x31

%backpropagation
for i = 1:max_step
    w1_prev = w1;
    w2_prev = w2;
    %feedforward
    z2 = w1 * a1';     %w1 = 30x65   a1 = 1303x65
    a2 = logsig(z2);   %a2 = 30x1303
    a2(31,:) = 1;      %a2 = 31x1303
    z3 = w2 * a2;     
    a3 = logsig(z3);    %a3 = 2x1303
    
    %compute gradient
    sigma3 = a3 - y_enc;        %sigma3 = 2x1303
    sigma2 = w2' * sigma3 .* (a2 .* (1-a2));   %sigma2 = 31x1303 
    grad1 = sigma2(2:31,:) * a1;            %grad1 = 30x65
    grad2 = sigma3 * a2';                   %grad2 = 2x31
    
    %update weights
    w1 = w1 - eta1 * grad1;
    w2 = w2 - eta2 * grad2;
    
    %calculate the cost
    term1 = -y_enc .* log(a3);
    term2 = (1 - y_enc) .* log(1-a3);
    cost(i,1) = sum(sum(term1 - term2));

    
    %if norm(w1_prev - w1) < 1e-20 && norm(w2_prev - w2) < 1e-20
        %break;
    %end
    
    
    
end

plot(cost)
axis([0 max_step 100 500])
end

