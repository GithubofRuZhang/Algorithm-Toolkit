function [label,Culster_data]=canopy(data,margin)
k = 0; 
[m,p]=size(data);
Z=zeros(m,1);
distance=squareform(pdist(data));
Y_center=[data zeros(m,1)];  
T2=margin*sum(mean(distance)); %自己设定的裕度值
Centra=zeros(20,p);
while size(Y_center,1)&& (k<20)
    k=k+1; 
    Centra(k,:)=Y_center(1,1:p);
    Y_center(1,:)=[];            
    L=size(Y_center,1); 
    if L
    distance1=(Y_center(:,1:p)-ones(L,1)*Y_center(1,1:p)).^2;   
    dist2=sum(distance1,2);   
    end
for i=1:L-1
    if(dist2(i)<T2)  %<T2说明是该类，在矩阵中删除
       Y_center(i,4)=1;
    end
end 
Y_center(Y_center(:,4)==1,:)=[];  
z=0;
end
%%
J_prev = inf; iter = 0; J = []; tol = 1e-3;
while true
    iter = iter + 1;
    distX = sum(data.^2, 2)*ones(1, k);                   
    distY = (sum(Centra(1:k,:).^2, 2)*ones(1, m))';
    distZ= -2*data*Centra(1:k,:)';
    dist=distX+distY+distZ;
    [a,label] = min(dist, [], 2) ;                    
    for i = 1:k
       Culster_data(i, :) = mean(data(label == i , :));              
    end
    J_cur = sum(sum((data - Culster_data(label, :)).^2, 2));         
    J = [J, J_cur];
    if norm(J_cur-J_prev, 'fro') < tol
        break;
    end
    J_prev = J_cur;
end
end
