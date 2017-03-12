function[para,n, errors] = perceptron(eta)
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here
data = importdata('TrainingSamplesDCT_8_new.mat');
FG = data.TrainsampleDCT_FG;
BG = data.TrainsampleDCT_BG;

para = zeros(1,65);
training_set = [FG;BG];
y = zeros(1303,1);
y(1:250,1) = 1;
y(251:1303,1) = -1;
errors = zeros(5000,1);
for n = 1:5000
    para_prev = para;
    for i = 1:1303
        %calculate the predicted value y_hat
        y_hat = para(1,2:65) * training_set(i,:)' + para(1,1);
        if y_hat < 0
            y_hat = -1;
        end
        if y_hat > 0
            y_hat = 1;
        end
        
        %run gradiant descent to update the parameters
        para(1,2:65) = para(1,2:65) + eta * (y(i,1) - y_hat) * training_set(i,:);
        para(1,1) = para(1,1) + eta * (y(i,1) - y_hat);
        
        if norm(y(i,1) - y_hat) ~= 0
            errors(n,1) = errors(n,1) + 1;
        end
    end
    

    
    if norm(para - para_prev) < 10e-6
        break;
    end
end

end

