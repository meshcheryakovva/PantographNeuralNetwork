% Конструктор GRNN-модели.

clear all
close all
clc

% Структура сети
PDelays1 = 30; % Длина линии задержки входа 1 (в том числе 0)
PDelays2 = 30; % Длина линии задержки входа 2 (в том числе 0)
PDelays3 = 5; % Длина линии задержки входа 3 (в том числе 0)

% Подготовка входных и выходных векторов для нейросети
load learndata.mat
TL = 2000; % длина временных рядов  для обучения
TM = 1000; % для моделирования
d = [1; diff(beg_op_num)]; % есть ли смена опоры
distop = zeros(1, PDelays1+TL+TM); % массив расстояния от опоры
for i = 2:PDelays1+TL+TM,
   if d(i) ~= 0, % смена опоры?
     distop(i)  = 0;
   else
     distop(i) = distop(i-1) + 0.25; % расстояние от опоры
   end
end

% данные - факторы для обучения
x1 = zeros(PDelays1 + PDelays2 + PDelays3, TL);
% данные - факторы для моделирования
x2 = zeros(PDelays1 + PDelays2 + PDelays3, TM);

% индексы входных векторов данных в каждом векторе-столбце из x
inp1 = 1:PDelays1;
inp2 = (PDelays1+1):(PDelays1+PDelays2);
inp3 = (PDelays1+PDelays2+1):(PDelays1+PDelays2+PDelays3);

% Данные для обучения
for j = 1:TL,
   x1(inp1, j) = h_nopkt(j:(PDelays1 + j - 1));
   x1(inp2, j) = h((PDelays1-PDelays2 + j):(PDelays1 + j - 1));
   x1(inp3, j) = distop((PDelays1-PDelays3 + j):(PDelays1 + j - 1));
end
y1 = pkt(PDelays1:(PDelays1+TL-1))';

% Данные для моделирования
for j = 1:TM,
   x2(inp1, j) = h_nopkt(j+TL:(PDelays1 + j - 1 + TL));
   x2(inp2, j) = h((PDelays1-PDelays2 + j + TL):(PDelays1 + j - 1 + TL));
   x2(inp3, j) = distop((PDelays1-PDelays3 + j + TL):(PDelays1 + j - 1 + TL));
end
y2 = pkt(PDelays1 + TL:(PDelays1+TL + TM - 1))';

% создание сети GRNN
spread = 0.1; % коэффициент перекрытия рад функций
net = newgrnn(x1, y1, spread);
view(net)

% выход сети на обучающей выборке
y1m = net(x1);
figure('Name', 'Выход сети на обучающей выборке')
plot(y1, '-b.')
hold on; grid on
plot(y1m, '-g.')
xlabel('наблюдение')
ylabel('Pkt, Н')
legend('Эксперимент', 'Обучение')
err1 = y1 - y1m; % ошибка моделирования
R2_1 = 1 - var(err1) / var(y1) % коэф-т дискриминации
e1 = mean(abs(err1)./ y1) % ср. отн. ошибка

% выход сети на выборке для прогнозирования
y2m = net(x2);
figure('Name', 'Выход сети на выборке для верификации')
plot(y2, '-b.')
hold on; grid on
plot(y2m, '-g.')
xlabel('наблюдение')
ylabel('Pkt, Н')
legend('Эксперимент', 'Моделирование')
err2 = y2 - y2m; % ошибка моделирования
R2_2 = 1 - var(err2) / var(y2) % коэф-т дискриминации
e2 = mean(abs(err2)./ y2) % ср. отн. ошибка

% СПЕКТРЫ
Fs = 1/0.25; % исходная частота дискретизации, 1/м
% метод Уэлча, односторонняя спектр. плотность
Pxx = pwelch(y2, 512, 64, 512, Fs, 'onesided');
PSD_y2 = Pxx;
% 1024,128,1024
Pxx = pwelch(y2m, 512, 64, 512, Fs, 'onesided');
PSD_y2m = Pxx;

Hpsd = dspdata.psd(Pxx(1:length(Pxx)), 'Fs', Fs);
FRQ = Hpsd.Frequencies;

figure('Name', 'Спектральные плотности мощности')
semilogy(FRQ, PSD_y2, 'b');
hold on; grid on
semilogy(FRQ, PSD_y2m, 'g');
xlabel('\omega, 1/м')
ylabel('S_P(\omega)')
legend('Эксперимент', 'Модель')



