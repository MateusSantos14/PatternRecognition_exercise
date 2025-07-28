clear; clc;

D=load('recfaces.dat');

Nr=50;  % No. de repeticoes

Ptrain=80; % Porcentagem de treinamento

tic; [STATS_0 TX_OK0 X0 m0 S0 posto0]=quadratico(D,Nr,Ptrain); Tempo0=toc;    % One COV matrix per class
tic; [STATS_1 TX_OK1 X1 m1 S1 posto1]=variante1(D,Nr,Ptrain,0.01); Tempo1=toc; % Regularization method 1 (Tikhonov)
tic; [STATS_2 TX_OK2 X2 m2 S2 posto2]=variante2(D,Nr,Ptrain); Tempo2=toc;     % One common COV matrix (pooled)
tic; [STATS_3 TX_OK3 X3 m3 S3 posto3]=variante3(D,Nr,Ptrain,0.5); Tempo3=toc; % Regularization method 2 (Friedman)
tic; [STATS_4 TX_OK4 X4 m4 S4 posto4]=variante4(D,Nr,Ptrain); Tempo4=toc;     % Naive Bayes Local (Based on quadratico)
tic; [STATS_5 TX_OK5 W]=linearMQ(D,Nr,Ptrain); Tempo5=toc;     % Classificador Linear de Minimos Quadrados

STATS_0
STATS_1
STATS_2
STATS_3
STATS_4
STATS_5

TEMPOS=[Tempo0 Tempo1 Tempo2 Tempo3 Tempo4 Tempo5]

boxplot([TX_OK0' TX_OK1' TX_OK2' TX_OK3' TX_OK4' TX_OK5'])
set(gca (), "xtick", [1 2 3 4 5 6 7 8 9], "xticklabel", {"Quadratico","Variante 1", "Variante 2","Variante 3","Variante 4","MQ"})
title('Conjunto Coluna');
xlabel('Classificador');
ylabel('Taxas de acerto');


