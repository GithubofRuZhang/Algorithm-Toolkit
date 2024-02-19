clc; clear; close all;
load('NNF_KMeans_SSA.mat')

data = G_out_data.data_process;    % Processed data, G_out_data.data_O is the data before processing
jiangwei_num = G_out_data.jiangwei_num;    % Dimensionality reduction number
Z = nnmf(data, jiangwei_num);     % NNMF (Non-negative Matrix Factorization)
data_get = tsne(data);  % t-SNE for high-dimensional data visualization, defaults to reducing to two dimensions
figure;
plot(data_get(:,1), data_get(:,2), '*', 'LineWidth', 1)
xlabel('x'); ylabel('y')
label_distance = G_out_data.label_distance;   % Distance metric for clustering
cluster_num = G_out_data.cluster_num;   % Number of clusters
[index, center] = kmeans(Z, cluster_num, 'Distance', label_distance);   % K-means clustering

sc_xishu = mean(silhouette(data, index'));   % Silhouette coefficient
a = unique(index); % Number of unique clusters
yang_num = length(a);
C = cell(1, length(a));
for i = 1:length(a)
     C(1, i) = {find(index == a(i))};
end
data1 = data;
color_list = G_out_data.color_list;   % Color palette
fu_str = G_out_data.fu_str;  % Marker style database 1
fu_str1 = G_out_data.fu_str1; % Marker style database 2
color_list_cha = G_out_data.color_list_cha;
color_all = G_out_data.color_all;
makesize = G_out_data.makesize;   % Marker size
Line_Width = G_out_data.LineWidth;   % Line width
data_num = size(data1, 2);   % Data dimensionality
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
zlabel1 = G_out_data.zlabel;   % zlabel (seems unused, possibly an error)
leg_kuang1 = G_out_data.leg_kuang;   % Legend frame
figure;
if yang_num <= 2
    N1 = 1; N2 = 2;
elseif yang_num <= 4 && yang_num > 2
    N1 = 2; N2 = 2;
else
    N2 = 3; N1 = ceil(yang_num / N2);
end
if(length(get_legend_str) < yang_num)
   for i = length(get_legend_str):yang_num
    get_legend_str{1, i} = ['Category ', num2str(i)];
  end
end
for NN = 1:yang_num
 subplot(N1, N2, NN)
 data_get = data1(C{1, NN}, :);
 for NN1 = 1:size(data_get, 1)
 plot(1:data_num, data_get(NN1, :), 'LineWidth', Line_Width(1), 'MarkerSize', makesize); hold on
 end

 set(gca, 'FontName', FontName1, 'FontSize', FontSize, 'LineWidth', kuang_with1)

 title(gca, title1)
 box(gca, kuang)
 grid(grid1)
 title(gca, get_legend_str{1, NN})
 xlabel(gca, xlabel1)
 ylabel(gca, ylabel1)
 ylim([min(min(data1)), max(max(data1))])
 xlim([0.5, data_num + 0.5])
end

figure;
rand_list1 = G_out_data.rand_list1;
for NN = 1:yang_num
 subplot(N1, N2, NN)
 data_get = data1(C{1, NN}, :);
 score = (mean(data_get)); score_L = (score - std(data_get)); score_H = (score + std(data_get));
 h1 = fill(gca, [1:length(score), fliplr(1:length(score))], [score_L, fliplr(score_H)], 'r'); hold(gca, 'on')
 h1.FaceColor = color_all(rand_list1(NN), :); % Fill color for the interval
 h1.EdgeColor = [1, 1, 1]; % Edge color set to white
 alpha(gca, 0.3)   % Set transparency
 plot(gca, 1:length(score), score, fu_str1{1, NN}, 'Color', color_all(rand_list1(NN), :), 'LineWidth', Line_Width(1), 'MarkerSize', makesize)
 set(gca, 'FontName', FontName1, 'FontSize', FontSize, 'LineWidth', kuang_with1)

 title(gca, title1)
 box(gca, kuang)
 grid(grid1)
 title(gca, get_legend_str{1, NN})
 xlabel(gca, xlabel1)
 ylabel(gca, ylabel1)
 ylim([min(min(data1)), max(max(data1))])
 xlim([0.5, data_num + 0.5])
end

cluster_max = G_out_data.cluster_max1;
sc_xishu1 = [];
for NN1 = 2:cluster_max
     [index, center] = kmeans(Z, NN1, 'Distance', label_distance);   % K-means clustering
     sc_xishu1(NN1 - 1) = mean(silhouette(data, index'));
end

disp_str = ['The silhouette coefficient for cluster numbers 2 to ', num2str(cluster_max), ' are respectively:'];
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
