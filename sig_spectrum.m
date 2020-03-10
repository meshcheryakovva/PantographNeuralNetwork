% —пектральный и автокоррел¤ционный анализ сигналов
clear all
close all
clc

load learndata.mat
L = 2048; % количество отсчетов

% ‘ормирование h_nopkt, h, d_op, pkt
h_nopkt = h_nopkt(1:L); % высота без нажати¤
h = h(1:L); % высота с нажатием

d = [1; diff(beg_op_num)]; % есть ли смена опоры
d_op = zeros(1, L); % рассто¤ние от опоры
for i = 1:L, % формируем сигналы линии задержки внешних входов
   if d(i) == 0, % нет смены опоры?
     d_op(i) = d_op(i-1) + 0.25; % рассто¤ние от опоры
   end
end

pkt = pkt(1:L);

l = 0.05:0.25:0.25*L;
figure('Name', '¬ходные сигналы h_nopkt(t) h(t)')
plot(l, h_nopkt, l, h, l, d_op)
figure('Name', '¬ыходной сигнал P(t)')
plot(l, pkt)


figure('Name', '¬ходные сигналы')
plot(h_nopkt, 'b')
hold on, grid on
plot(h, 'g')
plot(d_op, 'r')

figure('Name', 'P(dh)')
plot(h(1:250)-h_nopkt(1:250), pkt(1:250), '-b.')
hold on; grid on
plot(h(252:490)-h_nopkt(252:490), pkt(252:490), '-g.')
plot(h(495:745)-h_nopkt(495:745), pkt(495:745), '-r.')

% Ќелинейна¤ коррел¤ци¤?
n = 50; % количество отсчетов фрагментов сигнала
tau_max = 120; % максимальный сдвиг выходного сигнала относительно входа (в отсчетах)
%t0 = 30; % сдвиг в отсчетах относительно начала цикла
RHOk_hn = zeros(1, tau_max+1); % «начение критери¤  ендалла P(h_n)
PVALk_hn = zeros(1, tau_max+1); % соотв. веро¤тности нуль-гипотезы
RHOk_hp = zeros(1, tau_max+1); % «начение критери¤  ендалла P(h_p)
PVALk_hp = zeros(1, tau_max+1); % соотв. веро¤тности нуль-гипотезы
%RHOs_hn = zeros(1, tau_max+1); % «начение критери¤ —пирмена P(h_n)
%RHOs_hp = zeros(1, tau_max+1); % «начение критери¤ —пирмена P(h_p)

h1 = figure('Name', ' ритерий  ендалла P(h_n)');
h2 = figure('Name', ' ритерий  ендалла P(h_p)');
h11 = figure('Name', '¬еро¤тность дл¤  ритерий  ендалла P(h_n)');
h22 = figure('Name', '¬еро¤тность дл¤  ритерий  ендалла P(h_p)');

TAU = 0:tau_max;

for t0 = 740:10:850,
 for i = 0:tau_max,
  % «ависимость P(h_n)
  %  ритерий  ендалла
  [RHOk_hn(i+1), PVALk_hn(i+1)] = corr(h_nopkt((t0+1):(t0+n)), pkt((t0+1+i):(t0+n+i)), ...
      'type', 'Kendall');
  %  ритерий —пирмена P(h_n) 'Spearman'
 
  % «ависимость P(h_p)
  %  ритерий  ендалла
  [RHOk_hp(i+1), PVALk_hp(i+1)] = corr(h((t0+1):(t0+n)), pkt((t0+1+i):(t0+n+i)), ...
      'type', 'Kendall');
 end
 figure(h1)
 plot(TAU, RHOk_hn); grid on; hold on
 figure(h2);
 plot(TAU, RHOk_hp); grid on; hold on
 
 figure(h11)
 plot(TAU, PVALk_hn); grid on; hold on
 figure(h22);
 plot(TAU, PVALk_hp); grid on; hold on

end

%  ќЁ‘-“џ ј¬“ќ ќ––≈Ћя÷»»
figure('Name', 'ј ‘')
acf_h_nopkt = xcov(h_nopkt, 'coeff');
plot(acf_h_nopkt(ceil(end/2):ceil(end/2)+500),'b')
hold on, grid on
acf_h = xcov(h, 'coeff');
plot(acf_h(ceil(end/2):ceil(end/2)+500),'g')
acf_p = xcov(pkt, 'coeff');
plot(acf_p(ceil(end/2):ceil(end/2)+500),'r')
legend('K_h_n(\tau)', 'K_h_p(\tau)','K_P(\tau)')


% —ѕ≈ “–џ
Fs = 1/0.25; % исходна¤ частота дискретизации, 1/м
% метод ”элча, односторонн¤¤ спектр. плотность
Pxx = pwelch(h_nopkt,512,64,512,Fs,'onesided');
PSD_h_nopkt = Pxx;
% 1024,128,1024
Pxx = pwelch(h,512,64,512,Fs,'onesided');
PSD_h = Pxx;

Pxx = pwelch(pkt,512,64,512,Fs,'onesided');
PSD_pkt = Pxx;

Hpsd = dspdata.psd(Pxx(1:length(Pxx)),'Fs',Fs);
FRQ= Hpsd.Frequencies;

figure('Name', '—пектральные плотности мощности')
semilogy(FRQ, PSD_h_nopkt, 'b');
hold on; grid on
semilogy(FRQ, PSD_h, 'g');
semilogy(FRQ, PSD_pkt, 'r');
legend('S_h_n(\omega)', 'S_h_p(\omega)','S_P(\omega)')
