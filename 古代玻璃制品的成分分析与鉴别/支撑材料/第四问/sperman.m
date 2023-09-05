clear;clc
load data1.mat;
load data2.mat;
[R1,P1] = corr(data1,'type','Spearman');
[R2,P2] = corr(data2,'type','Spearman');
%figure_youwant
x1=tril(R1);%对高钾的相关系数进行正态分布检验
x2=tril(R2);
X1=x1(:);
X2=x2(:);
X1(find(X1==0))=[];
X1(find(X1==1))=[];
X2(find(X2==0))=[];
X2(find(X2==1))=[];
qqplot(X1);
qqplot(X2);

X1_excel=abs(X1);
X2_excel=abs(X2);