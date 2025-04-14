function processed_signal = signal_enhancement(noisy_signal, target_signal)
 
    n = length(noisy_signal);
    w = ones(1, n);
    b = zeros(1, n); 
    a1 = zeros(1, n);
    a2 = zeros(1, n);
    a3 = zeros(1, n);
    
    learn_rate = 0.1; 
    num_epochs = 500; 
    momentum = 0.9; 
    prev_grad_w = zeros(1, n); 
    prev_grad_b = zeros(1, n);     

    noisy_signal = (noisy_signal - min(noisy_signal)) / (max(noisy_signal) - min(noisy_signal)) * 100;
    target_signal = (target_signal - min(target_signal)) / (max(target_signal) - min(target_signal)) * 100;

    for epoch = 1:num_epochs
        for i = 1:n
            a1(i) = w(i) * noisy_signal(i);
            a2(i) = a1(i) + b(i);
            a3(i) = tanh(a2(i)); 
            
            error = target_signal(i) - a3(i);
            delta = error * (1 - a3(i)^2); 
            
            grad_w = delta * noisy_signal(i);
            grad_b = delta;
            
            w_update = learn_rate * grad_w + momentum * prev_grad_w(i);
            b_update = learn_rate * grad_b + momentum * prev_grad_b(i);
            
            w(i) = w(i) + w_update;
            b(i) = b(i) + b_update;
            
            prev_grad_w(i) = w_update;
            prev_grad_b(i) = b_update;
        end
    end
    
    processed_signal = a3;
    
    processed_signal = processed_signal * (max(target_signal) - min(target_signal)) / 100 + min(target_signal);
end
