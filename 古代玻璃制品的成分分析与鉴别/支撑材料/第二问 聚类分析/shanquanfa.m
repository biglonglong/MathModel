[a,b] = size(X);%一共有四个mat文件，其中的变量名均为X，故运行前请依次导入四个mat文件，mat文件请见支撑材料
%分别为Pb_Ba_without_weathering.mat，Pb_Ba_weathering.mat，High_K_without_weathering.mat，High_K_weathering.mat
H = zeros(1,b); 
for i = 1:b 
    v = X(:,i); 
    p = v / sum(v);
    e = -sum(p .* log(p)) / log(a);%e为信息熵
    H(i) = 1- e; %信息的效用值
end
W = H ./ sum(H);  % W为最后各化学成分指标的权重