clc;clear;close all;	
load('NNF_KMeans_POA.mat')	
	
data=G_out_data.data_process;    %总体处理后的数据 G_out_data.data_O  处理前的数据	
jiangwei_num=G_out_data.jiangwei_num;    %设置降维数	
 Z=nnmf(data,jiangwei_num);     %nnmf非负矩阵分解	
data_get=tsne(data);  %tsne 高维数据可视化，默认降成两维	
figure;	
 plot(data_get(:,1),data_get(:,2),'*','LineWidth',1)	
xlabel('x');ylabel('y')	
label_distance=G_out_data.label_distance;   %确定聚类采用距离	
cluster_num=G_out_data.cluster_num;   %确定聚类数	
[index,center] = kmeans(Z,cluster_num,'Distance',label_distance);   %Kmeans聚类	
	
sc_xishu=mean(silhouette(data,index'));   %轮廓系数	
a=unique(index); %找出分类出的个数	
yang_num=length(a);	
C=cell(1,length(a));	
for i=1:length(a)	
     C(1,i)={find(index==a(i))};	
end	
data1=Z;	
color_list=G_out_data.color_list;   %颜色数据库	
fu_str=G_out_data.fu_str;  %样式数据库1	
fu_str1=G_out_data.fu_str1; %样式数据库2	
color_list_cha=G_out_data.color_list_cha;	
color_all=G_out_data.color_all;	
makesize=G_out_data.makesize;   %标记大小	
Line_Width=G_out_data.LineWidth;   %线宽	
data_num=size(data1,2);   %数据维度	
legend_str=G_out_data.legend_str; 	
get_legend_str=strsplit(legend_str,',');	
	
FontSize=G_out_data.FontSize;   % 字体大小	
kuang_with1=G_out_data.kuang_width;   % 字体粗细	
FontName1=G_out_data.FontName;   % 字体样式	
xlabel1=G_out_data.xlabel;   % xlabel	
ylabel1=G_out_data.ylabel;   % ylabel	
title1=G_out_data.title;   % title	
kuang=G_out_data.kuang;   % 框的选择	
grid1=G_out_data.grid;   % 网格选择	
zlabel1=G_out_data.zlabel;   % ylabel	
leg_kuang1=G_out_data.leg_kuang;   % 图例框	
kuang_with1=G_out_data.kuang_width;   % 整体框设置	
 figure	
 for i=1:yang_num	
    data_get=data1(C{1,i},:);	
    plot3(data_get(:,1),data_get(:,2),data_get(:,3),fu_str{1,i},'Color',color_all(i,:),'LineWidth',Line_Width(1),'MarkerSize',makesize); hold on;	
 end	
	
	
 if(length(get_legend_str)<yang_num)	
    for i=length(get_legend_str):yang_num	
     get_legend_str{1,i}=['类别',num2str(i)];	
   end	
end	
	
set(gca,'FontName',FontName1,'FontSize',FontSize,'LineWidth',kuang_with1)	
	
	
 title(gca,title1)	
  box(gca,kuang)	
grid(grid1)	
 title(gca,title1)	
xlabel(gca,xlabel1)	
ylabel(gca,ylabel1)	
	
 legend(get_legend_str)	
 legend(leg_kuang1)	
	
	
	
	
	
	
	
	
	
	
	
	
cluster_max=G_out_data.cluster_max1;	
sc_xishu1=[];	
 for NN1=2:cluster_max	
      [index,center] = kmeans(Z,NN1,'Distance',label_distance);   %Kmeans聚类	
      sc_xishu1(NN1-1)=mean(silhouette(data,index'));	
 end	
	
	
	
	
 disp_str=['聚类数为2到',num2str(cluster_max),'轮廓系数分别为'];	
 disp(disp_str)	
 disp(sc_xishu1)	
 figure	
 yang_fu3={'--p','--o','-*','-+','-^','-p','-o','-x','-d','-s','-h'};	
 index_fu=randperm(length(yang_fu3),1);	
  plot(gca,2:cluster_max,sc_xishu1,yang_fu3{1,index_fu},'Color',color_all(randperm(length(color_all),1),:),'LineWidth',2,'MarkerSize',makesize)	
	
   xlabel('cluster-num')	
  ylabel('SC')	
  set(gca,'FontName',FontName1,'FontSize',FontSize,'LineWidth',kuang_with1)	
  title(gca,title1)	
  box(gca,kuang)	
  grid(grid1)	
