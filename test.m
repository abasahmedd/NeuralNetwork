clc;
clear all;

% عدد التكرارات للتدريب
epochs = 1000;

% إدخال القيم (مجموعة بيانات التدريب)
x = [0; 0.5; 1; 1.5; 2]; % المدخلات

y_actual = exp(-x); % القيم الصحيحة (لكن النموذج لن يراها مباشرة كمعادلة)

% الأوزان العشوائية الأولية
w = rand(1,1); % وزن واحد فقط لأننا نعمل على إدخال واحد لكل نيرون
b = rand(1); % انحياز عشوائي

% معدل التعلم
learnrate = 0.05;

% التدريب عبر الحلقات
for epoch = 1:epochs
    % الحساب الأمامي
    f = w * x + b; % ضرب الأوزان وإضافة الانحياز
    y2 = logsig(f); % تطبيق دالة التفعيل
    
    % حساب الخطأ
    error = y_actual - y2;
    
    % حساب مشتقة دالة التفعيل
    df = error .* (y2 .* (1 - y2));
    
    % تحديث الأوزان والانحياز
    w = w + learnrate * sum(df .* x);
    b = b + learnrate * sum(df);
    
    % طباعة الخطأ كل 100 تكرار
    if mod(epoch, 100) == 0
        disp(['Epoch ', num2str(epoch), ', Error: ', num2str(mean(abs(error)))]);
    end
end

% عرض النتائج النهائية
disp('الأوزان بعد التحديث:');
disp(w);

disp('الانحياز بعد التحديث:');
disp(b);

disp('القيم المتوقعة بعد التدريب:');
disp(y2);

disp('القيم الحقيقية:');
disp(y_actual);
