clc;clear all;
x = 0:0.2:10;
n = length(x);
w = linspace(0, 10, n); 
b = rand(1,n);
a1 = zeros(1, n); 
a2 = zeros(1, n);
a3 = zeros(1, n);
learn_ratae = 0.01 ;
target = (x>5);
for epoch = 1:100
    error_sum = 0 ;
    for i = 1 : n
        a1(i) = w(i)  *  x(i);
        a2(i) = a1(i) +  b(i);
        a3(i) = 1/(1+exp(-a2(i)));
        a4(i) = (exp(a2(i)) - exp(-a2(i))) / (exp(a2(i)) + exp(-a2(i)));
    error = target(i) - a3(i) ;
    grad_w = error * a3(i) * (1-a3(i))*x(i); 
    grad_b = error * a3(i) * (1-a3(i));
    w(i) = w(i) + learn_ratae * grad_w ;
    b(i) = b(i) + learn_ratae * grad_b ;
    error_sum = error_sum + error^2;
    end

end
s = 1:n;
plot(s,a1,'r',s,a2,'b',s,a3,'g',s,a4,'k');