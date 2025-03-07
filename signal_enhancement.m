function processed_signal = signal_enhancement(noisy_signal, target_signal)
    % المدخلات: noisy_signal - الإشارة المشوشة
    %          target_signal - الإشارة المستهدفة (الأصلية)
    % المخرجات: processed_signal - الإشارة المعالجة
    
    % تهيئة المتغيرات
    n = length(noisy_signal);
    w = ones(1, n); % الأوزان الابتدائية
    b = zeros(1, n); % الانحياز الابتدائي
    a1 = zeros(1, n);
    a2 = zeros(1, n);
    a3 = zeros(1, n);
    
    % تحسين المعلمات
    learn_rate = 0.1; % معدل تعلم أعلى لتسريع التقارب
    num_epochs = 500; % عدد عصور أكبر لتحسين التدريب
    momentum = 0.9; % إضافة زخم لتجنب التوقف في النقاط المحلية
    prev_grad_w = zeros(1, n); % متغير لتخزين التدرج السابق للأوزان
    prev_grad_b = zeros(1, n); % متغير لتخزين التدرج السابق للانحياز
    
    % تطبيع الإشارة المشوشة لتقليل تأثير القيم الكبيرة
    noisy_signal = (noisy_signal - min(noisy_signal)) / (max(noisy_signal) - min(noisy_signal)) * 100;
    target_signal = (target_signal - min(target_signal)) / (max(target_signal) - min(target_signal)) * 100;

    % التدريب
    for epoch = 1:num_epochs
        for i = 1:n
            % الحسابات الأمامية (Forward Pass)
            a1(i) = w(i) * noisy_signal(i);
            a2(i) = a1(i) + b(i);
            a3(i) = tanh(a2(i)); % استخدام Tanh بدلاً من Sigmoid لأداء أفضل مع الضوضاء
            
            % حساب الخطأ والتدرج
            error = target_signal(i) - a3(i);
            delta = error * (1 - a3(i)^2); % مشتقة Tanh
            
            % تحديث التدرج مع الزخم
            grad_w = delta * noisy_signal(i);
            grad_b = delta;
            
            % تحديث الأوزان والانحياز مع الزخم
            w_update = learn_rate * grad_w + momentum * prev_grad_w(i);
            b_update = learn_rate * grad_b + momentum * prev_grad_b(i);
            
            w(i) = w(i) + w_update;
            b(i) = b(i) + b_update;
            
            % حفظ التدرج للعصر التالي
            prev_grad_w(i) = w_update;
            prev_grad_b(i) = b_update;
        end
    end
    
    % إرجاع الإشارة المعالجة قبل إلغاء التطبيع
    processed_signal = a3;
    
    % إلغاء التطبيع للحصول على القيم في النطاق الأصلي
    processed_signal = processed_signal * (max(target_signal) - min(target_signal)) / 100 + min(target_signal);
end