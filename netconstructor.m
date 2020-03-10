% Конструктор нейросетевой модели. Сеть из 2 слоев.
% Загружать перед имит. моделированием.
% Сеть, задержанные сигналы - глобальные переменные

clear all
close all
clc

% Структура сети
% global net ADelays PDelays1 PDelays2 PDelays3 NumNeur
PDelays1 = 20; % Длина линии задержки входа 1 (в том числе 0)
PDelays2 = PDelays1; % Длина линии задержки входа 2 (в том числе 0)
PDelays3 = 3; % Длина линии задержки входа 3 (в том числе 0)
ADelays  = 3;   % Длина линии задержки обратной связи < PDelays1 !
NumNeur  = 50; % количество нейронов в скрытом слое

% Подготовка входных и выходных векторов для нейросети
load learndata.mat
TL = 500; % длина временных рядов  для обучения
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


for i = 1:PDelays1-1, % формируем сигналы линии задержки внешних входов
   Pi{1,i}  = h_nopkt(i); % для обучения
   Pi{2,i}  = h(i);
   Pi{3,i}  = distop(i);
   PiM{1,i}  = h_nopkt(TL+i); % для моделирования
   PiM{2,i}  = h(TL+i);
   PiM{3,i}  = distop(TL+i);
end

for i = 1:ADelays, % формируем сигналы линии задержки обратной связи
    Ai{1,i} = repmat(pkt(PDelays1-ADelays+i), NumNeur, 1);
    Ai{2,i} = pkt(PDelays1-ADelays+i);
    AiM{1,i} = repmat(pkt(PDelays1-ADelays+TL+i), NumNeur, 1);
    AiM{2,i} = pkt(PDelays1-ADelays+TL+i);
end

% Данные для обучения
for i = 1:TL,
   InputLearn{1,i}  = h_nopkt(PDelays1+i);
   InputLearn{2,i}  = h(PDelays1+i);
   InputLearn{3,i}  = distop(PDelays1+i);
   OutputLearn{1,i} = pkt(PDelays1+i);
end

% Данные для моделирования на основе обученной сети
for i = 1:TM,
   InputModel{1,i}  = h_nopkt(PDelays1+TL+i);
   InputModel{2,i}  = h(PDelays1+TL+i);
   InputModel{3,i}  = distop(PDelays1+TL+i);
   OutputModel{1,i} = pkt(PDelays1+TL+i);
end


% Конструктор сети (3 входа, 2 слоя, [biasConnect], [inputConnect], [layerConnect], [outputConnect], [targetConnect])
net = network(3, 2, [1; 1], [1 1 1; 0 0 0], [0 1; 1 0], [0 1], [0 1]); % конструктор класса
net.inputWeights{1,1}.delays = 0:1:(PDelays1-1); % формирование линии задержки для входа P1
net.inputWeights{1,2}.delays = 0:1:(PDelays2-1); % формирование линии задержки для входа P2
net.inputWeights{1,3}.delays = 0:1:(PDelays3-1); % формирование линии задержки для входа P3
net.layerWeights{1,2}.delays = 1:1:ADelays;      % формирование линии задержки для обр.связи

net.layers{1}.size = NumNeur; % количество нейронов в скрытом слое
net.layers{2}.size = 1;  % количество нейронов в выходном слое

net.inputs{1}.processFcns  = {'mapminmax', 'mapstd'}; % функции масштабирования входов
net.inputs{2}.processFcns  = {'mapminmax', 'mapstd'}; 
net.inputs{3}.processFcns  = {'mapminmax', 'mapstd'}; 
net.outputs{2}.processFcns = {'mapminmax', 'mapstd'}; % функции масштабирования выхода 2 слоя

net = configure(net, InputLearn, OutputLearn); % конфигурация, масштабирование входов/выходов

net.layers{1}.initFcn = 'initnw';      % функция инициализации 1 слоя (равномерное распределение весов по диапазону вх.сигналов)
net.layers{2}.initFcn = 'initnw';      % функция инициализации 2 слоя (для линейной ф. активации)
net.layers{1}.transferFcn = 'tansig';  % функция активации в скрытом слое
net.layers{2}.transferFcn = 'purelin'; % функция активации в выходном слое

net.initFcn  = 'initlay'; % инициализация сети через инициализацию слоев
net.performFcn = 'mse';  % критерий обучения - средний квадрат ошибки
net.plotFcns = {'plotperform'};

net.name = 'Модель контактной подвески';
net.inputs{1}.name = 'hn'; % Координата провода без нажатия
net.inputs{2}.name = 'hp'; % Координата провода при нажатии
net.inputs{3}.name = 'op'; % Расстояние от последней опоры
net.outputs{2}.name = 'ps'; % Усилие нажатия
net.layers{1}.name = 'Скрытый слой';
net.layers{2}.name = 'Выходной слой';

learnmode = 2; % режим обучения: 1 - адаптация, 2 - пакетное обучение

if learnmode == 1, % адаптация
   net.trainFcn = 'traincgp'; % метод сопр. градиентов
   net.adaptFcn = 'adaptwb'; % функция адаптации
   net.derivFcn = 'fpderiv'; % метод вычисления производных forward perturbation
else % пакетный режим
   net.trainFcn = 'traincgf'; % метод Флетчера-Ривса
   net.trainParam.min_step = 1e-9; % большее число шагов
   net.trainParam.epochs = 200; % ограничение на число итераций
%   net.trainFcn = 'trainlm'; % метод Левенберга-Маквардта
   net.derivFcn = 'bttderiv'; % метод вычисления производных Backpropagation through time
end

net = init(net);  % инициализация сети
net = closeloop(net); % формирование обратной связи
% view(net)

if learnmode == 1, % адаптация
   %[net,Y,E,Pf,Af,tr] = adapt(net, InputLearn, OutputLearn, Pi, Ai);
   [net,Y,E,Pf,Af,tr] = adapt(net, InputLearn, OutputLearn, Pi, Ai);

   figure
   plot(cell2mat(InputLearn)')
   figure
   plot(cell2mat(Y), 'b'), hold on, plot(cell2mat(OutputLearn), 'g')
else % пакетный режим
   net = train(net, InputLearn, OutputLearn, Pi, Ai);
   YL = net(InputLearn, Pi, Ai);
   YM = net(InputModel, PiM, AiM);
   
   figure('Name', 'Входные сигналы')
   plot(cell2mat(InputLearn)')
   grid on
   legend('Высота без нажатия', 'Высота с нажатием', 'Расстояние от опоры')
   figure('Name', 'Сила нажатия (обучение)')
   plot(cell2mat(YL), 'b'), hold on, plot(cell2mat(OutputLearn), 'g')
   grid on
   legend('Выход сети', 'Эксперим. данные')
   figure('Name', 'Сила нажатия (моделирование)')
   plot(cell2mat(YM), 'b'), hold on, plot(cell2mat(OutputModel), 'g')
   grid on
   legend('Выход сети', 'Эксперим. данные')
end


