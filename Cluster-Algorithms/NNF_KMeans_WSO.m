clc; clear; close all;    
load('NNF_KMeans_WSO.mat')    

data = G_out_data.data_process;    % Data after overall processing, G_out_data.data_O is the data before processing    
jiangwei_num = G_out_data.jiangwei_num;    % Setting the number of dimensions for dimensionality reduction    
Z = nnmf(data, jiangwei_num);     % Non-negative matrix factorization    
data_get = tsne(data);  % t-SNE visualization of high-dimensional data, default to reduce to two dimensions    
figure;    
plot(data_get(:, 1), data_get(:, 2), '*', 'LineWidth', 1)    
xlabel('x'); ylabel('y')    
label_distance = G_out_data.label_distance;   % Determine the distance for clustering    
cluster_num = G_out_data.cluster_num;   % Determine the number of clusters    
[index, center] = kmeans(Z, cluster_num, 'Distance', label_distance);   % Kmeans clustering    

sc_xishu = mean(silhouette(data, index'));   % Silhouette coefficient    
a = unique(index); % Find the number of classifications    
yang_num = length(a);    
C = cell(1, length(a));    
for i = 1:length(a)    
    C(1, i) = {find(index == a(i))};    
end    
data1 = Z;    
color_list = G_out_data.color_list;   % Color database    
fu_str = G_out_data.fu_str;  % Style database 1    
fu_str1 = G_out_data.fu_str1; % Style database 2    
color_list_cha = G_out_data.color_list_cha;    
color_all = G_out_data.color_all;    
makesize = G_out_data.makesize;   % Marker size    
Line_Width = G_out_data.LineWidth;   % Line width    
data_num = size(data1, 2);   % Data dimensions    
legend_str = G_out_data.legend_str;     
get_legend_str = strsplit(legend_str, ',');    

FontSize = G_out_data.FontSize;   % Font size    
kuang_with1 = G_out_data.kuang_width;   % Font weight    
FontName1 = G_out_data.FontName;   % Font style    
xlabel1 = G_out_data.xlabel;   % xlabel    
ylabel1 = G_out_data.ylabel;   % ylabel    
title1 = G_out_data.title;   % title    
kuang = G_out_data.kuang;   % Frame selection    
grid1 = G_out_data.grid;   % Grid selection    
zlabel1 = G_out_data.zlabel;   % ylabel    
leg_kuang1 = G_out_data.leg_kuang;   % Legend box    
kuang_with1 = G_out_data.kuang_width;   % Overall frame setting    

figure    
for i = 1:yang_num    
    data_get = data1(C{1, i}, :);    
    plot3(data_get(:, 1), data_get(:, 2), data_get(:, 3), fu_str{1, i}, 'Color', color_all(i, :), 'LineWidth', Line_Width(1), 'MarkerSize', makesize); hold on;    
end    

if (length(get_legend_str) < yang_num)    
    for i = length(get_legend_str):yang_num    
        get_legend_str{1, i} = ['Category', num2str(i)];    
    end    
end    

set(gca, 'FontName', FontName1, 'FontSize', FontSize, 'LineWidth', kuang_with1)    

title(gca, title1)    
box(gca, kuang)    
grid(grid1)    
xlabel(gca, xlabel1)    
ylabel(gca, ylabel1)    
legend(get_legend_str)    
legend(leg_kuang1)    

cluster_max = G_out_data.cluster_max1;    
sc_xishu1 = [];    
for NN1 = 2:cluster_max    
    [index, center] = kmeans(Z, NN1, 'Distance', label_distance);   % Kmeans clustering    
    sc_xishu1(NN1 - 1) = mean(silhouette(data, index'));    
end    

disp_str = ['Number of clusters from 2 to ', num2str(cluster_max), ', silhouette coefficients are:'];    
disp(disp_str)    
disp(sc_xishu1)    

figure    
yang_fu3 = {'--p', '--o', '-*', '-+', '-^', '-p', '-o', '-x', '-d', '-s', '-h'};    
index_fu = randperm(length(yang_fu3), 1);    
plot(gca, 2:cluster_max, sc_xishu1, yang_fu3{1, index_fu}, 'Color', color_all(randperm(length(color_all), 1), :), 'LineWidth', 2, 'MarkerSize', makesize)    
xlabel('cluster-num')    
ylabel('SC')    
set(gca, 'FontName', FontName1, 'FontSize', FontSize, 'LineWidth', kuang_with1)    
title(gca, title1)    
box(gca, kuang)    
grid(grid1)
