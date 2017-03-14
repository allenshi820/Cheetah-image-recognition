
count = 1;
for max_step = 1000:100:20000
    [w1, w2] = MLP(0.005, 0.0005, max_step);
    [p_err, ~] = classification(w1,w2);
    perr(count,1) = p_err;
    count = count + 1;
end

plot(perr);