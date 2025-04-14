clc;
clear all;

epochs = 1000;

x = [0; 0.5; 1; 1.5; 2];

y_actual = exp(-x);

w = rand(1,1);
b = rand(1);

learnrate = 0.05;

for epoch = 1:epochs
    f = w * x + b; 
    y2 = logsig(f); 
    
    error = y_actual - y2;
    
    df = error .* (y2 .* (1 - y2));
    
    w = w + learnrate * sum(df .* x);
    b = b + learnrate * sum(df);
    
    if mod(epoch, 100) == 0
        disp(['Epoch ', num2str(epoch), ', Error: ', num2str(mean(abs(error)))]);
    end
end

