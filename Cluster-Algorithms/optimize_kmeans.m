function [Mdl,fMin,Convergence_curve]  = optimize_kmeans(data_O,data_Z,num_pop,num_iter,method_mti,cluster_limits)  
   % data_O原数据集   data_Z降维后数据集
     pop=num_pop;
     M=num_iter;
     LB=[cluster_limits(1),1];
     UB=[cluster_limits(2),3];
     nvars=length(LB);
     fit_fitrensemble1=@fit_kmeans;
    if strcmp( method_mti,'SSA麻雀搜索算法')==1
       [fMin,Mdl,Convergence_curve,pos]=SSA_SSA(pop,M,LB,UB,nvars,fit_fitrensemble1,data_O,data_Z);      
     elseif strcmp( method_mti,'DBO蜣螂优化算法')==1
       [fMin,Mdl,Convergence_curve,pos]=DBO_DBO(pop,M,LB,UB,nvars,fit_fitrensemble1,data_O,data_Z);      
    elseif strcmp( method_mti,'SA模拟退火算法')==1 
        [fMin,Mdl,Convergence_curve,pos]=SA_SA(pop,M,LB,UB,nvars,fit_fitrensemble1,data_O,data_Z);       
    elseif strcmp( method_mti,'PSO粒子群算法')==1 
        [fMin,Mdl,Convergence_curve,pos]=PSO_PSO(pop,M,LB,UB,nvars,fit_fitrensemble1,data_O,data_Z);         
    elseif strcmp( method_mti,'SCA正余弦优化算法')==1       
    [fMin,Mdl,Convergence_curve,pos]=SCA_SCA(pop,M,LB,UB,nvars,fit_fitrensemble1,data_O,data_Z);         
     elseif strcmp( method_mti,'POA鹈鹕优化算法')==1 
    [fMin,Mdl,Convergence_curve,pos]=POA_POA(pop,M,LB,UB,nvars,fit_fitrensemble1,data_O,data_Z);    
       elseif strcmp( method_mti,'GWO灰狼优化算法')==1
      [fMin,Mdl,Convergence_curve,pos]=GWO_GWO(pop,M,LB,UB,nvars,fit_fitrensemble1,data_O,data_Z );  
     elseif strcmp( method_mti,'IGWO改进灰狼优化算法')==1
     [fMin,Mdl,Convergence_curve,pos]=IGWO_IGWO(pop,M,LB,UB,nvars,fit_fitrensemble1,data_O,data_Z );      
   elseif strcmp( method_mti,'AVOA非洲秃鹰优化算法')==1
     [fMin,Mdl,Convergence_curve,pos]=AVOA_AVOV(pop,M,LB,UB,nvars,fit_fitrensemble1,data_O,data_Z );      
  elseif strcmp( method_mti,'CSA变色龙优化算法')==1  
     [fMin,Mdl,Convergence_curve,pos]=CSA_CAS(pop,M,LB,UB,nvars,fit_fitrensemble1,data_O,data_Z);        
 elseif strcmp( method_mti,'GTO大猩猩优化算法')==1  
     [fMin,Mdl,Convergence_curve,pos]=GTO_GTO(pop,M,LB,UB,nvars,fit_fitrensemble1,data_O,data_Z);       
 elseif strcmp( method_mti,'NGO北方苍鹰优化算法')==1  
     [fMin,Mdl,Convergence_curve,pos]=NGO_NGO(pop,M,LB,UB,nvars,fit_fitrensemble1,data_O,data_Z);        
 elseif strcmp( method_mti,'SO蛇优化算法')==1  
     [fMin,Mdl,Convergence_curve,pos]=SO_SO(pop,M,LB,UB,nvars,fit_fitrensemble1,data_O,data_Z);       
  elseif strcmp( method_mti,'WSO白鲨优化算法')==1    
     [fMin,Mdl,Convergence_curve,pos]=WSO_WSO(pop,M,LB,UB,nvars,fit_fitrensemble1,data_O,data_Z);             
  end
       figure
     plot(-Convergence_curve,'--p','LineWidth',1.2,'Color',[160,123,194]./255) 
     xticks([1:length(Convergence_curve)])
     title('optimize process')
     xlabel('iter')
     ylabel('fitness')
     grid on
   set(gca,"FontName","Times New Roman","FontSize",12,"LineWidth",1.2)
   box off 
  distance_str={'sqeuclidean','cityblock','cosine'};
  disp([method_mti, '优化 kmeans:   ',"cluster_num:",num2str(round(pos(1))),'  distance: ',distance_str{1,round(pos(2))}]) 
end
function [fitness_value,index]=fit_kmeans(pop,data_O,data_Z)
     distance_str={'sqeuclidean','cityblock','cosine'};
     [index,~] = kmeans(data_Z,round(pop(1)),'Distance',distance_str{1,round(pop(2))});  
    fitness_value=-mean(silhouette(data_O,index'));
end

function [fMin,Mdl_best,Convergence_curve,bestX]=SSA_SSA(pop,M,LB,UB,nvars,fobj,data_O,data_Z)       
 P_percent = 0.2;    % The population size of producers accounts for "P_percent" percent of the total population size       
   %生产者占所有种群的0.2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
pNum = round( pop *  P_percent );    % The population size of the producers    
%生产者数量取整
dim=nvars;
lb= LB.*ones( 1,dim );    % Lower limit/bounds/     a vector    约束上限
ub= UB.*ones( 1,dim );    % Upper limit/bounds/     a vector  约束下限
%Initialization
for i = 1 : pop    
    x( i, : ) = lb + (ub - lb) .* rand( 1, dim );   %随机初始化n个种群
    x(i, : ) = Bounds( x(i, : ),lb, ub);
    [fit( i ),model{1,i}] = fobj( x( i, : ),data_O,data_Z ) ;               %计算所有群体的适应情况，如果求最小值的,越小代表适应度越好      
end
% disp(1111)
% 以下找到最小值对应的麻雀群
pFit = fit;                      
pX = x;                            % The individual's best position corresponding to the pFit
[ fMin, bestI ] = min( fit );      % fMin denotes the global optimum fitness value
bestX = x( bestI, : );             % bestX denotes the global optimum position corresponding to fMin
Mdl_best=model{1,bestI};
 % Start updating the solutions.
for t = 1 : M         
  [ ~, sortIndex ] = sort( pFit );% Sort.
     
  [fmax,B]=max( pFit );
   worse= x(B,:);         %找到最差的个体      

   r2=rand(1);      %产生随机数    感觉没有啥科学依据，就是随机数
   %大概意思就是在0.8概率内原来种群乘一个小于1的数，种群整体数值缩小了
   %大概意思就是在0.2概率内原来种群乘一个小于1的数，种群整体数值+1
   %变化后种群的数值还是要限制在约束里面
   %对前pNum适应度最好的进行变化 ，即生产者进行变化，可见生产者是挑最好的
if(r2<0.8)
    for i = 1 : pNum                                                        % Equation (3)
         r1=rand(1);
        x( sortIndex( i ), : ) = pX( sortIndex( i ), : )*exp(-(i)/(r1*M)); %将种群按适应度排序后更新 
        x( sortIndex( i ), : ) = Bounds( x( sortIndex( i ), : ), lb, ub );  %将种群限制在约束范围内

         [fit( sortIndex( i ) ),model{1,sortIndex( i )}]= fobj(x( sortIndex( i ), : ),data_O,data_Z ) ;
    end
  else
  for i = 1 : pNum            
  x( sortIndex( i ), : ) = pX( sortIndex( i ), : )+randn(1)*ones(1,dim);
  x( sortIndex( i ), : ) = Bounds( x( sortIndex( i ), : ), lb, ub );
  [fit( sortIndex( i ) ),model{1,sortIndex( i )}]= fobj(x( sortIndex( i ), : ),data_O,data_Z ) ;

  end      
end
 %把经过变化后最好的种群记录下来  
 [ ~, bestII ] = min( fit );      
  bestXX = x( bestII, : );            
%   disp(2222)
 %下面是乞讨者 
   for i = ( pNum + 1 ) : pop                     % Equation (4)
         A=floor(rand(1,dim)*2)*2-1;           %产生1和-1的随机数
         
          if( i>(pop/2))
           %如果i>种群的一半，代表遍历到适应度靠后的一段，代表这些序列的种群可能在挨饿
           x( sortIndex(i ), : )=randn(1)*exp((worse-pX( sortIndex( i ), : ))/(i)^2);  
           %适应度不好的，即靠后的麻雀乘了一个大于1的数，向外拓展
          else
        x( sortIndex( i ), : )=bestXX+(abs(( pX( sortIndex( i ), : )-bestXX)))*(A'*(A*A')^(-1))*ones(1,dim);  
           %这是适应度出去介于生产者之后，又在种群的前半段的,去竞争生产者的食物，在前面最好种群的基础上
           %再进行变化一次，在原来的基础上减一些值或者加一些值
         end  
        x( sortIndex( i ), : ) = Bounds( x( sortIndex( i ), : ), lb, ub );  %更新后种群的限制在变量范围

       [fit( sortIndex( i ) ),model{1,sortIndex( i )}]= fobj(x( sortIndex( i ), : ),data_O,data_Z ) ;

   end
   %在全部种群中找可以意识到危险的麻雀
  c=randperm(numel(sortIndex));
   b=sortIndex(c(1:round(0.2*length(c))));
  for j =  1  : length(b)      % Equation (5)
    if( pFit( sortIndex( b(j) ) )>(fMin) )
         %如果适应度比最开始最小适应度差的话，就在原来的最好种群上增长一部分值
        x( sortIndex( b(j) ), : )=bestX+(randn(1,dim)).*(abs(( pX( sortIndex( b(j) ), : ) -bestX)));
        
        else
        %如果适应度达到开始最小的适应度值，就在原来的最好种群上随机增长或减小一部分
        x( sortIndex( b(j) ), : ) =pX( sortIndex( b(j) ), : )+(2*rand(1)-1)*(abs(pX( sortIndex( b(j) ), : )-worse))/ ( pFit( sortIndex( b(j) ) )-fmax+1e-50);

          end
        x( sortIndex(b(j) ), : ) = Bounds( x( sortIndex(b(j) ), : ), lb, ub );

        [fit( sortIndex( b(j) ) ),model{1,sortIndex( b(j) )}]= fobj(x( sortIndex( b(j) ), : ),data_O,data_Z ) ;

 end
    for i = 1 : pop 
        %如果哪个种群适应度好了，就把变化的替换掉原来的种群
        if ( fit( i ) < pFit( i ) )
            pFit( i ) = fit( i );
            pX( i, : ) = x( i, : );
        end
        
        if( pFit( i ) < fMin )   %最优值以及最优值位置看是否变化
           fMin= pFit( i );
            bestX = pX( i, : );
            Mdl_best=model{1,i};
         
            
        end
    end
    Convergence_curve(t)=fMin;
end
end
%%
%
function [fMin , Mdl_best, Convergence_curve,bestX ] =  DBO_DBO(pop,M,c,d,dim,fobj,data_O,data_Z) 
        
   P_percent = 0.2;    % The population size of producers accounts for "P_percent" percent of the total population size       


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
pNum = round( pop *  P_percent );    % The population size of the producers   


lb= c.*ones( 1,dim );    % Lower limit/bounds/     a vector
ub= d.*ones( 1,dim );    % Upper limit/bounds/     a vector
%Initialization
for i = 1 : pop
    
    x( i, : ) = lb + (ub - lb) .* rand( 1, dim );  
    x(i, : ) = Bounds( x(i, : ),lb, ub);
    [fit( i ),model{1,i}]= fobj( x( i, : ),data_O,data_Z  ) ;                       
end

pFit = fit;                       
pX = x; 
 XX=pX;    
[ fMin, bestI ] = min( fit );      % fMin denotes the global optimum fitness value
bestX = x( bestI, : );             % bestX denotes the global optimum position corresponding to fMin
Mdl_best=model{1,bestI};
 % Start updating the solutions.
for t = 1 : M    
       
        [fmax,B]=max(fit);
        worse= x(B,:);   
       r2=rand(1);
 
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for i = 1 : pNum    
        if(r2<0.9)
            r1=rand(1);
          a=rand(1,1);
          if (a>0.1)
           a=1;
          else
           a=-1;
          end
    x( i , : ) =  pX(  i , :)+0.3*abs(pX(i , : )-worse)+a*0.1*(XX( i , :)); % Equation (1)
       else
            
           aaa= randperm(180,1);
           if ( aaa==0 ||aaa==90 ||aaa==180 )
            x(  i , : ) = pX(  i , :);   
           end
         theta= aaa*pi/180;   
       
       x(  i , : ) = pX(  i , :)+tan(theta).*abs(pX(i , : )-XX( i , :));    % Equation (2)      

        end
      
        x(  i , : ) = Bounds( x(i , : ), lb, ub );    
        [fit( i ),model{1,i}]= fobj( x( i, : ),data_O,data_Z  ) ; 
    
    end 
 [ fMMin, bestII ] = min( fit );      % fMin denotes the current optimum fitness value
  bestXX = x( bestII, : );             % bestXX denotes the current optimum position 
Mdl_best=model{1,bestII};
 R=1-t/M;                           %
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 Xnew1 = bestXX.*(1-R); 
     Xnew2 =bestXX.*(1+R);                    %%% Equation (3)
   Xnew1= Bounds( Xnew1, lb, ub );
   Xnew2 = Bounds( Xnew2, lb, ub );
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
     Xnew11 = bestX.*(1-R); 
     Xnew22 =bestX.*(1+R);                     %%% Equation (5)
   Xnew11= Bounds( Xnew11, lb, ub );
    Xnew22 = Bounds( Xnew22, lb, ub );
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
    for i = ( pNum + 1 ) : round( pop *  0.4 )                  % Equation (4)
     x( i, : )=bestXX+((rand(1,dim)).*(pX( i , : )-Xnew1)+(rand(1,dim)).*(pX( i , : )-Xnew2));
   x(i, : ) = Bounds( x(i, : ), Xnew1, Xnew2 );
    [fit( i ),model{1,i}]= fobj( x( i, : ),data_O,data_Z  ) ; 
%   [fit( i ),model{1,i}] = fobj( x( i, : ),data_O,data_Z ) ;    
   end
   
  for i = round( pop *  0.4 )+1: round( pop *  0.6 )                  % Equation (6)

   
        x( i, : )=pX( i , : )+((randn(1)).*(pX( i , : )-Xnew11)+((rand(1,dim)).*(pX( i , : )-Xnew22)));
       x(i, : ) = Bounds( x(i, : ),lb, ub);
        [fit( i ),model{1,i}]= fobj( x( i, : ),data_O,data_Z  ) ; 
%        [fit( i ),model{1,i}]= fobj( x( i, : ),data_O,data_Z ) ;    
  
  end
  
  for j = round( pop *  0.6 )+1 : pop                 % Equation (7)
       x( j,: )=bestX+randn(1,dim).*((abs(( pX(j,:  )-bestXX)))+(abs(( pX(j,:  )-bestX))))./2;
      x(j, : ) = Bounds( x(j, : ), lb, ub );
       [fit( j ),model{1,j}]= fobj( x( j, : ),data_O,data_Z  ) ; 
%  [fit( j ),model{1,j}] = fobj( x(j, : ),data_O,data_Z ) ;    
  end
   % Update the individual's best fitness vlaue and the global best fitness value
     XX=pX;
    for i = 1 : pop 
        if ( fit( i ) < pFit( i ) )
            pFit( i ) = fit( i );
            pX( i, : ) = x( i, : );
        end
        
        if( pFit( i ) < fMin )
           % fMin= pFit( i );
           fMin= pFit( i );
            bestX = pX( i, : );
            Mdl_best=model{1,i};
          %  a(i)=fMin;
            
        end
    end
  
     Convergence_curve(t)=fMin;
  
     
end
end
%%

function [Destination_fitness,Mdl_best,Convergence_curve,Destination_position]=SCA_SCA(N,Max_iteration,lb,ub,dim,fobj,data_O,data_Z)         
% [Destination_fitness,Mdl_best,Convergence_curve]=SCA_SCA(N,Max_iteration,lb,ub,dim,fit_fitrensemble1,data_O,data_Z);         
% display('SCA is optimizing your problem');

%Initialize the set of random solutions
% X=initialization(N,dim,ub,lb);
for i = 1 : N
    
    X( i, : ) = Bounds(lb + (ub - lb) .* rand( 1, dim ) ,lb,ub ) ;    
  
end

Destination_position=zeros(1,dim);
Destination_fitness=inf;

Convergence_curve=zeros(1,Max_iteration);
Objective_values = zeros(1,size(X,1));

% Calculate the fitness of the first set and find the best one
for i=1:size(X,1)
%     =fobj(X(i,:));
%  [fit( j ),model{1,j}]= fobj( x( j, : ),data_O,data_Z  ) ; 
    [Objective_values(1,i),model{1,i}]= fobj( X( i, : ),data_O,data_Z);
    if i==1
        Destination_position=X(i,:);
        Destination_fitness=Objective_values(1,i);
    elseif Objective_values(1,i)<Destination_fitness
        Destination_position=X(i,:);
        Destination_fitness=Objective_values(1,i);
    end
    
    All_objective_values(1,i)=Objective_values(1,i);
end
[~,index_choose]=min(Objective_values);
Mdl_best=model{1,index_choose};
%Main loop
t=2; % start from the second iteration since the first iteration was dedicated to calculating the fitness
while t<=Max_iteration
    
    % Eq. (3.4)
    a = 2;
    Max_iteration = Max_iteration;
    r1=a-t*((a)/Max_iteration); % r1 decreases linearly from a to 0
    
    % Update the position of solutions with respect to destination
    for i=1:size(X,1) % in i-th solution
        for j=1:size(X,2) % in j-th dimension
            
            % Update r2, r3, and r4 for Eq. (3.3)
            r2=(2*pi)*rand();
            r3=2*rand;
            r4=rand();
            
            % Eq. (3.3)
            if r4<0.5
                % Eq. (3.1)
                X(i,j)= X(i,j)+(r1*sin(r2)*abs(r3*Destination_position(j)-X(i,j)));
                X(i,:)= Bounds(X(i,:),lb,ub );  
               
            else
                % Eq. (3.2)
                X(i,j)= X(i,j)+(r1*cos(r2)*abs(r3*Destination_position(j)-X(i,j)));
                X(i,:)= Bounds(X(i,:),lb,ub );  
            end
            
        end
    end
    
    for i=1:size(X,1)
         
        % Check if solutions go outside the search spaceand bring them back
        Flag4ub=X(i,:)>ub;
        Flag4lb=X(i,:)<lb;
        X(i,:)=(X(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;
        X(i,:)= Bounds(X(i,:),lb,ub );  
        
        % Calculate the objective values
         [Objective_values(1,i),model{1,i}]= fobj( X( i, : ),data_O,data_Z );
        
        % Update the destination if there is a better solution
        if Objective_values(1,i)<Destination_fitness
            Destination_position=X(i,:);
            Destination_fitness=Objective_values(1,i);
             Mdl_best=model{1,i};
        end
    end
    
    Convergence_curve(t)=Destination_fitness;
    
    % Display the iteration and best optimum obtained so far
%     if mod(t,50)==0
%         display(['At iteration ', num2str(t), ' the optimum is ', num2str(Destination_fitness)]);
%     end
    
    % Increase the iteration counter
    t=t+1;
end
end
%%
function[Best_score,Mdl_best,POA_curve,Best_pos]=POA_POA(SearchAgents,Max_iterations,lowerbound,upperbound,dimension,fobj,data_O,data_Z)

lowerbound=ones(1,dimension).*(lowerbound);                              % Lower limit for variables
upperbound=ones(1,dimension).*(upperbound);                              % Upper limit for variables

% INITIALIZATION
for i=1:dimension
    X(:,i) = lowerbound(i)+rand(SearchAgents,1).*(upperbound(i) - lowerbound(i));                          % Initial population
end

for i =1:SearchAgents
%     L=X(i,:);
     [fit(i),model{1,i}]= fobj( X( i, : ),data_O,data_Z);
%     fit(i)=fobj(L);
end
%
[~,index_nedd]=min(fit);
Mdl_best=model{1,index_nedd};
for t=1:Max_iterations
    % update the best condidate solution
    [best , location]=min(fit);
    if t==1
        Xbest=X(location,:);                                           % Optimal location
        fbest=best;                                           % The optimization objective function
    elseif best<fbest
        fbest=best;
        Xbest=X(location,:);
    end
    
    % UPDATE location of food
    
    X_FOOD=[];
    k=randperm(SearchAgents,1);
    X_FOOD=X(k,:);
    F_FOOD=fit(k);
    
    %%
    for i=1:SearchAgents
        
        % PHASE 1: Moving towards prey (exploration phase)
        I=round(1+rand(1,1));
        if fit(i)> F_FOOD
            X_new=X(i,:)+ rand(1,1).*(X_FOOD-I.* X(i,:)); %Eq(4)
        else
            X_new=X(i,:)+ rand(1,1).*(X(i,:)-1.*X_FOOD); %Eq(4)
        end
        X_new= max(X_new,lowerbound);X_new = min(X_new,upperbound);
        
        % Updating X_i using (5)
          [f_new,model_new]= fobj(X_new,data_O,data_Z);
%         f_new = fobj(X_new);
        if f_new <= fit (i)
            X(i,:) = X_new;
            fit (i)=f_new;
            Mdl_best=model_new;
        end
        % END PHASE 1: Moving towards prey (exploration phase)
        
        % PHASE 2: Winging on the water surface (exploitation phase)
        X_new=X(i,:)+0.2*(1-t/Max_iterations).*(2*rand(1,dimension)-1).*X(i,:);% Eq(6)
        X_new= max(X_new,lowerbound);X_new = min(X_new,upperbound);
        
        % Updating X_i using (7)
%         f_new = fobj(X_new);
%         if f_new <= fit (i)
%             X(i,:) = X_new;
%             fit (i)=f_new;
%         end
      % Updating X_i using (7)
        [f_new,model_new]= fobj(X_new,data_O,data_Z);
%         f_new = fobj(X_new);
        if f_new <= fit (i)
            X(i,:) = X_new;
            fit (i)=f_new;
            Mdl_best=model_new;
        end
        % END PHASE 2: Winging on the water surface (exploitation phase)
    end

    best_so_far(t)=fbest;
    average(t) = mean (fit);
    
end
Best_score=fbest;
Best_pos=Xbest;
POA_curve=best_so_far;
end
%%
function [Alpha_score,Mdl_best,Convergence_curve,Alpha_pos]=GWO_GWO(SearchAgents_no,Max_iter,lb,ub,dim,fobj,data_O,data_Z)
 
% initialize alpha, beta, and delta_pos
Alpha_pos=zeros(1,dim);
Alpha_score=inf; %change this to -inf for maximization problems

Beta_pos=zeros(1,dim);
Beta_score=inf; %change this to -inf for maximization problems

Delta_pos=zeros(1,dim);
Delta_score=inf; %change this to -inf for maximization problems

%Initialize the positions of search agents
Positions=initialization(SearchAgents_no,dim,ub,lb);

Convergence_curve=zeros(1,Max_iter);

l=0;% Loop counter

% Main loop
while l<Max_iter
    for i=1:size(Positions,1)  
        
       % Return back the search agents that go beyond the boundaries of the search space
        Flag4ub=Positions(i,:)>ub;
        Flag4lb=Positions(i,:)<lb;
        Positions(i,:)=(Positions(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;               
        
        % Calculate objective function for each search agent
%           fitness=fobj(Positions(i,:));
         [fitness1(i),model{1,i}]= fobj(Positions(i,:),data_O,data_Z);
         fitness=fitness1(i);
%         Mdl_best=model{1,1};
        % Update Alpha, Beta, and Delta
        if fitness<Alpha_score 
            Alpha_score=fitness; % Update alpha
            Alpha_pos=Positions(i,:);
        end
        
        if fitness>Alpha_score && fitness<Beta_score 
            Beta_score=fitness; % Update beta
            Beta_pos=Positions(i,:);
        end
        
        if fitness>Alpha_score && fitness>Beta_score && fitness<Delta_score 
            Delta_score=fitness; % Update delta
            Delta_pos=Positions(i,:);
        end

    end
    
    
    a=2-l*((2)/Max_iter); % a decreases linearly fron 2 to 0
    
    % Update the Position of search agents including omegas
    for i=1:size(Positions,1)
        for j=1:size(Positions,2)     
                       
            r1=rand(); % r1 is a random number in [0,1]
            r2=rand(); % r2 is a random number in [0,1]
            
            A1=2*a*r1-a; % Equation (3.3)
            C1=2*r2; % Equation (3.4)
            
            D_alpha=abs(C1*Alpha_pos(j)-Positions(i,j)); % Equation (3.5)-part 1
            X1=Alpha_pos(j)-A1*D_alpha; % Equation (3.6)-part 1
                       
            r1=rand();
            r2=rand();
            
            A2=2*a*r1-a; % Equation (3.3)
            C2=2*r2; % Equation (3.4)
            
            D_beta=abs(C2*Beta_pos(j)-Positions(i,j)); % Equation (3.5)-part 2
            X2=Beta_pos(j)-A2*D_beta; % Equation (3.6)-part 2       
            
            r1=rand();
            r2=rand(); 
            
            A3=2*a*r1-a; % Equation (3.3)
            C3=2*r2; % Equation (3.4)
            
            D_delta=abs(C3*Delta_pos(j)-Positions(i,j)); % Equation (3.5)-part 3
            X3=Delta_pos(j)-A3*D_delta; % Equation (3.5)-part 3             
            
            Positions(i,j)=(X1+X2+X3)/3;% Equation (3.7)
            
        end
    end
    l=l+1;    
    Convergence_curve(l)=Alpha_score;
end
[~,indx_best]=min(fitness1);
Mdl_best= model{1,indx_best};
%   [fitness1(i),model{1,i}]
end
%%
% [Alpha_score,Mdl_best,Convergence_curve,Alpha_pos]=GWO_GWO(SearchAgents_no,Max_iter,lb,ub,dim,fobj,data_O,data_Z )
function [Alpha_score,Mdl_best,Convergence_curve,Alpha_pos]=IGWO_IGWO(N,Max_iter,lb,ub,dim,fobj,data_O,data_Z)

lu = [lb .* ones(1, dim); ub .* ones(1, dim)];

% Initialize alpha, beta, and delta positions
Alpha_pos=zeros(1,dim);
Alpha_score=inf; %change this to -inf for maximization problems

Beta_pos=zeros(1,dim);
Beta_score=inf; %change this to -inf for maximization problems

Delta_pos=zeros(1,dim);
Delta_score=inf; %change this to -inf for maximization problems

% Initialize the positions of wolves
Positions=initialization(N,dim,ub,lb);
Positions = Bounds (Positions, lb, ub);

% Calculate objective function for each wolf
for i=1:size(Positions,1)
%     Fit(i) = fobj(Positions(i,:));
    [Fit(i),model{1,i}]= fobj(Positions(i,:),data_O,data_Z );
end

% Personal best fitness and position obtained by each wolf
pBestScore = Fit;
pBest = Positions;

neighbor = zeros(N,N);
Convergence_curve=zeros(1,Max_iter);
iter = 0;% Loop counter

%% Main loop
while iter < Max_iter
    for i=1:size(Positions,1)
        fitness = Fit(i);
        
        % Update Alpha, Beta, and Delta
        if fitness<Alpha_score
            Alpha_score=fitness; % Update alpha
            Alpha_pos=Positions(i,:);
        end
        
        if fitness>Alpha_score && fitness<Beta_score
            Beta_score=fitness; % Update beta
            Beta_pos=Positions(i,:);
        end
        
        if fitness>Alpha_score && fitness>Beta_score && fitness<Delta_score
            Delta_score=fitness; % Update delta
            Delta_pos=Positions(i,:);
        end
    end
    
    %% Calculate the candiadate position Xi-GWO
    a=2-iter*((2)/Max_iter); % a decreases linearly from 2 to 0
    
    % Update the Position of search agents including omegas
    for i=1:size(Positions,1)
        for j=1:size(Positions,2)
            
            r1=rand(); % r1 is a random number in [0,1]
            r2=rand(); % r2 is a random number in [0,1]
            
            A1=2*a*r1-a;                                    % Equation (3.3)
            C1=2*r2;                                        % Equation (3.4)
            
            D_alpha=abs(C1*Alpha_pos(j)-Positions(i,j));    % Equation (3.5)-part 1
            X1=Alpha_pos(j)-A1*D_alpha;                     % Equation (3.6)-part 1
            
            r1=rand();
            r2=rand();
            
            A2=2*a*r1-a;                                    % Equation (3.3)
            C2=2*r2;                                        % Equation (3.4)
            
            D_beta=abs(C2*Beta_pos(j)-Positions(i,j));      % Equation (3.5)-part 2
            X2=Beta_pos(j)-A2*D_beta;                       % Equation (3.6)-part 2
            
            r1=rand();
            r2=rand();
            
            A3=2*a*r1-a;                                    % Equation (3.3)
            C3=2*r2;                                        % Equation (3.4)
            
            D_delta=abs(C3*Delta_pos(j)-Positions(i,j));    % Equation (3.5)-part 3
            X3=Delta_pos(j)-A3*D_delta;                     % Equation (3.5)-part 3
            
            X_GWO(i,j)=(X1+X2+X3)/3;                        % Equation (3.7)
            
        end
        X_GWO(i,:) = Bounds(X_GWO(i,:), lb, ub);
         [Fit_GWO(i),model{1,i}]= fobj(X_GWO(i,:),data_O,data_Z );
%         Fit_GWO(i) = fobj(X_GWO(i,:));
    end
    
    % Calculate the candiadate position Xi-DLH
    radius = pdist2(Positions, X_GWO, 'euclidean');         % Equation (10)
    dist_Position = squareform(pdist(Positions));
    r1 = randperm(N,N);
    
    for t=1:N
        neighbor(t,:) = (dist_Position(t,:)<=radius(t,t));
        [~,Idx] = find(neighbor(t,:)==1);                   % Equation (11)             
        random_Idx_neighbor = randi(size(Idx,2),1,dim);
        
        for d=1:dim
            X_DLH(t,d) = Positions(t,d) + rand .*(Positions(Idx(random_Idx_neighbor(d)),d)...
                - Positions(r1(t),d));                      % Equation (12)
        end
        X_DLH(t,:) = Bounds(X_DLH(t,:), lb, ub);
         [  Fit_DLH(t) ,model{1,t}]= fobj(X_DLH(t,:),data_O,data_Z);
%         Fit_DLH(t) = fobj(X_DLH(t,:));
    end
    
    % Selection  
    tmp = Fit_GWO < Fit_DLH;                                % Equation (13)
    tmp_rep = repmat(tmp',1,dim);
    
    tmpFit = tmp .* Fit_GWO + (1-tmp) .* Fit_DLH;
    tmpPositions = tmp_rep .* X_GWO + (1-tmp_rep) .* X_DLH;
    
    % Updating
    tmp = pBestScore <= tmpFit;                             % Equation (13)
    tmp_rep = repmat(tmp',1,dim);
    
    pBestScore = tmp .* pBestScore + (1-tmp) .* tmpFit;
    pBest = tmp_rep .* pBest + (1-tmp_rep) .* tmpPositions;
    
    Fit = pBestScore;
    Positions = pBest;
    
    %
    iter = iter+1;
    neighbor = zeros(N,N);
    Convergence_curve(iter) = Alpha_score;  
end
[~,best_index]=min(Fit_DLH(t));
Mdl_best=model{1,best_index};
end

%%
function [Best_vulture1_F,Mdl_best,convergence_curve,Best_vulture1_X]=AVOA_AVOV(pop_size,max_iter,lower_bound,upper_bound,variables_no,fobj,data_O,data_Z)

    % initialize Best_vulture1, Best_vulture2
    Best_vulture1_X=zeros(1,variables_no);
    Best_vulture1_F=inf;
    Best_vulture2_X=zeros(1,variables_no);
    Best_vulture2_F=inf;

    %Initialize the first random population of vultures
    X=initialization(pop_size,variables_no,upper_bound,lower_bound);

    %  Controlling parameter
    p1=0.6;
    p2=0.4;
    p3=0.6;
    alpha=0.8;
    betha=0.2;
    gamma=2.5;

    %%Main loop
    current_iter=0; % Loop counter

    while current_iter < max_iter
        for i=1:size(X,1)
            % Calculate the fitness of the population
            current_vulture_X = X(i,:);
               [current_vulture_F,Mdl_best1]= fobj(current_vulture_X,data_O,data_Z);
%             current_vulture_F=fobj(current_vulture_X);
                if i==1
                   Mdl_best=Mdl_best1;
                end
            % Update the first best two vultures if needed
            if current_vulture_F<Best_vulture1_F
                Best_vulture1_F=current_vulture_F; % Update the first best bulture
                Best_vulture1_X=current_vulture_X;
                Mdl_best=Mdl_best1;
            end
            if current_vulture_F>Best_vulture1_F && current_vulture_F<Best_vulture2_F
                Best_vulture2_F=current_vulture_F; % Update the second best bulture
                Best_vulture2_X=current_vulture_X;
            end
        end

        a=unifrnd(-2,2,1,1)*((sin((pi/2)*(current_iter/max_iter))^gamma)+cos((pi/2)*(current_iter/max_iter))-1);
        P1=(2*rand+1)*(1-(current_iter/max_iter))+a;

        % Update the location
        for i=1:size(X,1)
            current_vulture_X = X(i,:);  % pick the current vulture back to the population
            F=P1*(2*rand()-1);  

            random_vulture_X=random_select(Best_vulture1_X,Best_vulture2_X,alpha,betha);
            
            if abs(F) >= 1 % Exploration:
                current_vulture_X = exploration(current_vulture_X, random_vulture_X, F, p1, upper_bound, lower_bound);
            elseif abs(F) < 1 % Exploitation:
                current_vulture_X = exploitation(current_vulture_X, Best_vulture1_X, Best_vulture2_X, random_vulture_X, F, p2, p3, variables_no, upper_bound, lower_bound);
            end

            X(i,:) = current_vulture_X; % place the current vulture back into the population
             X(i,:)  = Bounds(X(i,:) , lower_bound, upper_bound);
        end

        current_iter=current_iter+1;
        convergence_curve(current_iter)=Best_vulture1_F;

       

%         fprintf("In Iteration %d, best estimation of the global optimum is %4.4f \n ", current_iter,Best_vulture1_F );
    end

end

%
function [current_vulture_X] = exploitation(current_vulture_X, Best_vulture1_X, Best_vulture2_X, ...
                                                                      random_vulture_X, F, p2, p3, variables_no, upper_bound, lower_bound)

% phase 1
    if  abs(F)<0.5
        if rand<p2
            A=Best_vulture1_X-((Best_vulture1_X.*current_vulture_X)./(Best_vulture1_X-current_vulture_X.^2))*F;
            B=Best_vulture2_X-((Best_vulture2_X.*current_vulture_X)./(Best_vulture2_X-current_vulture_X.^2))*F;
            current_vulture_X=(A+B)/2;
        else
            current_vulture_X=random_vulture_X-abs(random_vulture_X-current_vulture_X)*F.*levyFlight(variables_no);
        end
    end
    % phase 2
    if  abs(F)>=0.5
        if rand<p3
            current_vulture_X=(abs((2*rand)*random_vulture_X-current_vulture_X))*(F+rand)-(random_vulture_X-current_vulture_X);
        else
            s1=random_vulture_X.* (rand()*current_vulture_X/(2*pi)).*cos(current_vulture_X);
            s2=random_vulture_X.* (rand()*current_vulture_X/(2*pi)).*sin(current_vulture_X);
            current_vulture_X=random_vulture_X-(s1+s2);
        end
    end
end
%
function [current_vulture_X] = exploration(current_vulture_X, random_vulture_X, F, p1, upper_bound, lower_bound)

    if rand<p1
        current_vulture_X=random_vulture_X-(abs((2*rand)*random_vulture_X-current_vulture_X))*F;
    else
        current_vulture_X=(random_vulture_X-(F)+rand()*((upper_bound-lower_bound)*rand+lower_bound));
    end
    
end
%
function [random_vulture_X]=random_select(Best_vulture1_X,Best_vulture2_X,alpha,betha)

    probabilities=[alpha, betha ];
    
    if (rouletteWheelSelection( probabilities ) == 1)
            random_vulture_X=Best_vulture1_X;
    else
            random_vulture_X=Best_vulture2_X;
    end

end
%
function [ o ]=levyFlight(d)
  
    beta=3/2;

    sigma=(gamma(1+beta)*sin(pi*beta/2)/(gamma((1+beta)/2)*beta*2^((beta-1)/2)))^(1/beta);
    u=randn(1,d)*sigma;
    v=randn(1,d);
    step=u./abs(v).^(1/beta);

    o=step;

end
%
% function [ X ] = boundaryCheck(X, lb, ub)
% 
%     for i=1:size(X,1)
%             FU=X(i,:)>ub;
%             FL=X(i,:)<lb;
%             X(i,:)=(X(i,:).*(~(FU+FL)))+ub.*FU+lb.*FL;
%     end
% end
function [index] = rouletteWheelSelection(x)
    index=find(rand() <= cumsum(x) ,1,'first');
end
%%
% [Best_vulture1_F,Mdl_best,convergence_curve,Best_vulture1_X]=AVOA_AVOV(pop_size,max_iter,lower_bound,upper_bound,variables_no,fobj,data_O,data_Z)
function [fmin0,Mdl_best,cg_curve,gPosition]=CSA_CAS(searchAgents,iteMax,lb,ub,dim,fobj,data_O,data_Z)

%%%%* 1
if size(ub,2)==1
    ub=ones(1,dim)*ub;
    lb=ones(1,dim)*lb;
end

% Convergence curve
cg_curve=zeros(1,iteMax);

% Initial population

chameleonPositions=initialization(searchAgents,dim,ub,lb);% Generation of initial solutions 

% Evaluate the fitness of the initial population

fit=zeros(searchAgents,1);


for i=1:searchAgents
     [fit(i,1),Model{1,i}]= fobj(chameleonPositions(i,:),data_O,data_Z);
%      fit(i,1)=fobj(chameleonPositions(i,:));
end

% Initalize the parameters of CSA
fitness=fit; % Initial fitness of the random positions
 
[fmin0,index]=min(fit);

chameleonBestPosition = chameleonPositions; % Best position initialization
gPosition = chameleonPositions(index,:); % initial global position
Mdl_best=Model{1,index};

v=0.1*chameleonBestPosition;% initial velocity
 
v0=0.0*v;

% Start CSA 
% Main parameters of CSA
rho=1.0;
p1=2.0;  
p2=2.0;  
c1=2.0; 
c2=1.80;  
gamma=2.0; 
alpha = 4.0;  
beta=3.0; 
 

 % Start CSA
for t=1:iteMax
a = 2590*(1-exp(-log(t))); 
omega=(1-(t/iteMax))^(rho*sqrt(t/iteMax)) ; 
p1 = 2* exp(-2*(t/iteMax)^2);  % 
p2 = 2/(1+exp((-t+iteMax/2)/100)) ;
        
mu= gamma*exp(-(alpha*t/iteMax)^beta) ;

ch=ceil(searchAgents*rand(1,searchAgents));
% Update the position of CSA (Exploration)
for i=1:searchAgents  
             if rand>=0.1
                  chameleonPositions(i,:)= chameleonPositions(i,:)+ p1*(chameleonBestPosition(ch(i),:)-chameleonPositions(i,:))*rand()+... 
                     + p2*(gPosition -chameleonPositions(i,:))*rand();
             else 
                 for j=1:dim
                   chameleonPositions(i,j)=   gPosition(j)+mu*((ub(j)-lb(j))*rand+lb(j))*sign(rand-0.50) ;
                 end 
              end   
end       
 % Rotation of the chameleons - Update the position of CSA (Exploitation)
 
 %  % Chameleon velocity updates and find a food source
     for i=1:searchAgents
               
        v(i,:)= omega*v(i,:)+ p1*(chameleonBestPosition(i,:)-chameleonPositions(i,:))*rand +.... 
               + p2*(gPosition-chameleonPositions(i,:))*rand;        

         chameleonPositions(i,:)=chameleonPositions(i,:)+(v(i,:).^2 - v0(i,:).^2)/(2*a);
     end
    
  v0=v;
  
 % handling boundary violations
 for i=1:searchAgents
     if chameleonPositions(i,:)<lb
        chameleonPositions(i,:)=lb;
     elseif chameleonPositions(i,:)>ub
            chameleonPositions(i,:)=ub;
     end
 end
 
 % Relocation of chameleon positions (Randomization) 
for i=1:searchAgents
    
    ub_=sign(chameleonPositions(i,:)-ub)>0;   
    lb_=sign(chameleonPositions(i,:)-lb)<0;
       
    chameleonPositions(i,:)=(chameleonPositions(i,:).*(~xor(lb_,ub_)))+ub.*ub_+lb.*lb_;  %%%%%*2
 
  [fit(i,1),Model{1,i}]= fobj(chameleonPositions(i,:),data_O,data_Z);
      
      if fit(i)<fitness(i)
                 chameleonBestPosition(i,:) = chameleonPositions(i,:); % Update the best positions  
                 fitness(i)=fit(i); % Update the fitness
                 model2{1,i}=Model{1,i};
      end
 end


% Evaluate the new positions

[fmin,index]=min(fitness); % finding out the best positions  

% Updating gPosition and best fitness
if fmin < fmin0
    gPosition = chameleonBestPosition(index,:); % Update the global best positions
    Mdl_best=model2{1,index};
    fmin0 = fmin;
end

% Visualize the results
   cg_curve(t)=fmin0; % Best found value until iteration t

end
% ngPosition=find(fitness== min(fitness)); 
% g_best=chameleonBestPosition(ngPosition(1),:);  % Solutin of the problem
% fmin0 =fobj(g_best);
end
%%
% [fmin0,Mdl_best,cg_curve,gPosition]=CSA_CAS(searchAgents,iteMax,lb,ub,dim,fobj,data_O,data_Z)
function [Silverback_Score,Mdl_best,convergence_curve,Silverback]=GTO_GTO(pop_size,max_iter,lower_bound,upper_bound,variables_no,fobj,data_O,data_Z)

% initialize Silverback
Silverback=[];
Silverback_Score=inf;

%Initialize the first random population of Gorilla
X=initialization(pop_size,variables_no,upper_bound,lower_bound);


convergence_curve=zeros(max_iter,1);

for i=1:pop_size   
%     Pop_Fit(i)=fobj(X(i,:));%#ok
     [Pop_Fit(i),Model{1,i}]= fobj(X(i,:),data_O,data_Z);
      Mdl_best=Model{1,1};
     if Pop_Fit(i)<Silverback_Score 
            Silverback_Score=Pop_Fit(i); 
            Silverback=X(i,:);
            Mdl_best=Model{1,i};
    end
end


GX=X(:,:);
lb=ones(1,variables_no).*lower_bound; 
ub=ones(1,variables_no).*upper_bound; 

%  Controlling parameter

p=0.03;
Beta=3;
w=0.8;

%%Main loop
for It=1:max_iter 
    
    a=(cos(2*rand)+1)*(1-It/max_iter);
    C=a*(2*rand-1); 

% Exploration:

    for i=1:pop_size
        if rand<p    
            GX(i,:) =(ub-lb)*rand+lb;
        else  
            if rand>=0.5
                Z = unifrnd(-a,a,1,variables_no);
                H=Z.*X(i,:);   
                GX(i,:)=(rand-a)*X(randi([1,pop_size]),:)+C.*H; 
            else   
                GX(i,:)=X(i,:)-C.*(C*(X(i,:)- GX(randi([1,pop_size]),:))+rand*(X(i,:)-GX(randi([1,pop_size]),:))); %ok ok 
               
            end
             
        end
        GX(i,:) = Bounds(GX(i,:), lower_bound, upper_bound);
    end       
       

    
    % Group formation operation 
    for i=1:pop_size
         [New_Fit,Model1]= fobj(GX(i,:),data_O,data_Z);
%          New_Fit= fobj(GX(i,:));          
         if New_Fit<Pop_Fit(i)
            Pop_Fit(i)=New_Fit;
            X(i,:)=GX(i,:);
            Mdl_best=Model1;
         end
         if New_Fit<Silverback_Score 
            Silverback_Score=New_Fit; 
            Silverback=GX(i,:);
         end
    end
    
% Exploitation:  
    for i=1:pop_size
       if a>=w  
            g=2^C;
            delta= (abs(mean(GX)).^g).^(1/g);
            GX(i,:)=C*delta.*(X(i,:)-Silverback)+X(i,:); 
       else
           
           if rand>=0.5
              h=randn(1,variables_no);
           else
              h=randn(1,1);
           end
           r1=rand; 
           GX(i,:)= Silverback-(Silverback*(2*r1-1)-X(i,:)*(2*r1-1)).*(Beta*h); 
           
       end
       GX(i,:) = Bounds(GX(i,:), lower_bound, upper_bound);
    end
   
%     GX = Bounds(GX, lower_bound, upper_bound);
    
    % Group formation operation    
    for i=1:pop_size
          [New_Fit,Model1]= fobj(GX(i,:),data_O,data_Z);
         if New_Fit<Pop_Fit(i)
            Pop_Fit(i)=New_Fit;
            X(i,:)=GX(i,:);
            Mdl_best=Model1;
         end
         if New_Fit<Silverback_Score 
            Silverback_Score=New_Fit; 
            Silverback=GX(i,:);
         end
    end
             
convergence_curve(It)=Silverback_Score;
% fprintf("In Iteration %d, best estimation of the global optimum is %4.4f \n ", It,Silverback_Score );
         
end 
end
%%
function [Score,Mdl_best,NGO_curve,Best_pos]=NGO_NGO(Search_Agents,Max_iterations,Lowerbound,Upperbound,dimensions,fobj,data_O,data_Z)
% tic
% disp('PLEASE WAIT, The program is running.')
Lowerbound=ones(1,dimensions).*(Lowerbound);                              % Lower limit for variables
Upperbound=ones(1,dimensions).*(Upperbound);                              % Upper limit for variables

X=[];
X_new=[];
fit=[];
fit_new=[];
NGO_curve=zeros(1,Max_iterations);

%
for i=1:dimensions
    X(:,i) = Lowerbound(i)+rand(Search_Agents,1).*(Upperbound(i) -Lowerbound(i));              % Initial population
end
for i =1:Search_Agents
%     L=X(i,:);
    [fit(i),Model{1,i}]= fobj(X(i,:),data_O,data_Z);
%     fit(i)=fobj(L);                    % Fitness evaluation (Explained at the top of the page. )
end

for t=1:Max_iterations  % algorithm iteration    
    %  update: BEST proposed solution
    [best , blocation]=min(fit);
    
    if t==1
        xbest=X(blocation,:);                                           % Optimal location
        fbest=best;  
        % The optimization objective function
        Mdl_best=Model{1,blocation};
    elseif best<fbest
        fbest=best;
        xbest=X(blocation,:);
        Mdl_best=Model{1,blocation};
    end
    
    
    % UPDATE Northern goshawks based on PHASE1 and PHASE2
    
    for i=1:Search_Agents
        % Phase 1: Exploration
        I=round(1+rand);
        k=randperm(Search_Agents,1);
        P=X(k,:); % Eq. (3)
        F_P=fit(k);
        
        if fit(i)> F_P
            X_new(i,:)=X(i,:)+rand(1,dimensions) .* (P-I.*X(i,:)); % Eq. (4)
        else
            X_new(i,:)=X(i,:)+rand(1,dimensions) .* (X(i,:)-P); % Eq. (4)
        end
        X_new(i,:) = max(X_new(i,:),Lowerbound);X_new(i,:) = min(X_new(i,:),Upperbound);
        
        % update position based on Eq (5)
%         L=X_new(i,:);
%         fit_new(i)=fobj(L);
         [fit_new(i),Model1{1,i}]= fobj(X_new(i,:),data_O,data_Z );
        if(fit_new(i)<fit(i))
            X(i,:) = X_new(i,:);
            fit(i) = fit_new(i);
            Model{1,i}=Model1{1,i};
        end
        % END PHASE 1
        
        % PHASE 2 Exploitation
        R=0.02*(1-t/Max_iterations);% Eq.(6)
        X_new(i,:)= X(i,:)+ (-R+2*R*rand(1,dimensions)).*X(i,:);% Eq.(7)
        
        X_new(i,:) = max(X_new(i,:),Lowerbound);X_new(i,:) = min(X_new(i,:),Upperbound);
        
        % update position based on Eq (8)
         [fit_new(i),Model1{1,i}]= fobj(X_new(i,:),data_O,data_Z);
        if(fit_new(i)<fit(i))
            X(i,:) = X_new(i,:);
            fit(i) = fit_new(i);
            Model{1,i}=Model1{1,i};
        end
        % END PHASE 2
        
    end% end for i=1:N
    
    %
    % SAVE BEST SCORE
    best_so_far(t)=fbest; % save best solution so far
    average(t) = mean (fit);
    Score=fbest;
    Best_pos=xbest;
    NGO_curve(t)=Score;
end
%
end
%%
function [fval,Mdl_best,gbest_t,Xfood] = SO_SO(N,T, lb,ub,dim,fobj,data_O,data_Z)

%initial 
vec_flag=[1,-1];
Threshold=0.25;
Thresold2= 0.6;
C1=0.5;
C2=.05;
C3=2;
if length(lb)<2
X=lb+rand(N,dim)*(ub-lb);
else 
X=repmat(lb,N,1)+rand(N,dim).*repmat((ub-lb),N,1);
end
for i=1:N
     [fitness(i),Model1{1,i}]= fobj(X(i,:),data_O,data_Z);
%  fitness(i)=fobj(X(i,:));   
end
[GYbest, gbest] = min(fitness);
Xfood = X(gbest,:);
Mdl_best=Model1{1,gbest};
%Diving the swarm into two equal groups males and females
Nm=round(N/2);%eq.(2&3)
Nf=N-Nm;
Xm=X(1:Nm,:);
Xf=X(Nm+1:N,:);
fitness_m=fitness(1:Nm);
fitness_f=fitness(Nm+1:N);
[fitnessBest_m, gbest1] = min(fitness_m);
Xbest_m = Xm(gbest1,:);
[fitnessBest_f, gbest2] = min(fitness_f);
Xbest_f = Xf(gbest2,:);
for t = 1:T
    Temp=exp(-((t)/T));  %eq.(4)
  Q=C1*exp(((t-T)/(T)));%eq.(5)
    if Q>1        Q=1;    end
    % Exploration Phase (no Food)
if Q<Threshold
    for i=1:Nm
        for j=1:1:dim
            rand_leader_index = floor(Nm*rand()+1);
            X_randm = Xm(rand_leader_index, :);
            flag_index = floor(2*rand()+1);
            Flag=vec_flag(flag_index);
            Am=exp(-fitness_m(rand_leader_index)/(fitness_m(i)+eps));%eq.(7)
            Xnewm(i,j)=X_randm(j)+Flag*C2*Am*((ub(1)-lb(1))*rand+lb(1));%eq.(6)
        end
    end
    for i=1:Nf
        for j=1:1:dim
            rand_leader_index = floor(Nf*rand()+1);
            X_randf = Xf(rand_leader_index, :);
            flag_index = floor(2*rand()+1);
            Flag=vec_flag(flag_index);
            Af=exp(-fitness_f(rand_leader_index)/(fitness_f(i)+eps));%eq.(9)
            Xnewf(i,j)=X_randf(j)+Flag*C2*Af*((ub(1)-lb(1))*rand+lb(1));%eq.(8)
        end
    end
else %Exploitation Phase (Food Exists)
    if Temp>Thresold2  %hot
        for i=1:Nm
            flag_index = floor(2*rand()+1);
            Flag=vec_flag(flag_index);
            for j=1:1:dim
                Xnewm(i,j)=Xfood(j)+C3*Flag*Temp*rand*(Xfood(j)-Xm(i,j));%eq.(10)
            end
        end
        for i=1:Nf
            flag_index = floor(2*rand()+1);
            Flag=vec_flag(flag_index);
            for j=1:1:dim
                Xnewf(i,j)=Xfood(j)+Flag*C3*Temp*rand*(Xfood(j)-Xf(i,j));%eq.(10)
            end
        end
    else %cold
        if rand>0.6 %fight
            for i=1:Nm
                for j=1:1:dim
                    FM=exp(-(fitnessBest_f)/(fitness_m(i)+eps));%eq.(13)
                    Xnewm(i,j)=Xm(i,j) +C3*FM*rand*(Q*Xbest_f(j)-Xm(i,j));%eq.(11)
                    
                end
            end
            for i=1:Nf
                for j=1:1:dim
                    FF=exp(-(fitnessBest_m)/(fitness_f(i)+eps));%eq.(14)
                    Xnewf(i,j)=Xf(i,j)+C3*FF*rand*(Q*Xbest_m(j)-Xf(i,j));%eq.(12)
                end
            end
        else%mating
            for i=1:Nf
                for j=1:1:dim
                    Mm=exp(-fitness_f(i)/(fitness_m(i)+eps));%eq.(17)
                    Xnewm(i,j)=Xm(i,j) +C3*rand*Mm*(Q*Xf(i,j)-Xm(i,j));%eq.(15
                end
            end
            for i=1:Nf
                for j=1:1:dim
                    Mf=exp(-fitness_m(i)/(fitness_f(i)+eps));%eq.(18)
                    Xnewf(i,j)=Xf(i,j) +C3*rand*Mf*(Q*Xm(i,j)-Xf(i,j));%eq.(16)
                end
            end
            flag_index = floor(2*rand()+1);
            egg=vec_flag(flag_index);
            if egg==1
                [GYworst, gworst] = max(fitness_m);
                Xnewm(gworst,:)=lb+rand*(ub-lb);%eq.(19)
                [GYworst, gworst] = max(fitness_f);
                Xnewf(gworst,:)=lb+rand*(ub-lb);%eq.(20)
            end
        end
    end
end
    for j=1:Nm
         Flag4ub=Xnewm(j,:)>ub;
         Flag4lb=Xnewm(j,:)<lb;
        Xnewm(j,:)=(Xnewm(j,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;
        [y,Model2]= fobj( Xnewm(j,:),data_O,data_Z);
%         y = fobj(Xnewm(j,:));
        if y<fitness_m(j)
            fitness_m(j)=y;
            Model1{1,j}=Model2;
            Xm(j,:)= Xnewm(j,:);
          
        end
    end
    
    [Ybest1,gbest1] = min(fitness_m);
    Mdl_best=Model1{1,gbest1};
    for j=1:Nf
         Flag4ub=Xnewf(j,:)>ub;
         Flag4lb=Xnewf(j,:)<lb;
        Xnewf(j,:)=(Xnewf(j,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;
               [y,Model2]= fobj( Xnewf(j,:),data_O,data_Z);
%         y = fobj(Xnewm(j,:));
        if y<fitness_f(j)
            fitness_f(j)=y;
            Xf(j,:)= Xnewf(j,:);
            Model1{1,j}=Model2;
        end
    end
    
    [Ybest2,gbest2] = min(fitness_f);
%     disp(gbest2)
%     disp(Model1{1,gbest2});
    Mdl_best=Model1{1,gbest2};

    if Ybest1<fitnessBest_m
        Xbest_m = Xm(gbest1,:);
        fitnessBest_m=Ybest1;
    end
    if Ybest2<fitnessBest_f
        Xbest_f = Xf(gbest2,:);
        fitnessBest_f=Ybest2;
        
    end
    if Ybest1<Ybest2
        gbest_t(t)=min(Ybest1);
    else
        gbest_t(t)=min(Ybest2);
        
    end
    if fitnessBest_m<fitnessBest_f
        GYbest=fitnessBest_m;
        Xfood=Xbest_m;
    else
        GYbest=fitnessBest_f;
        Xfood=Xbest_f;
    end
    
end
fval = GYbest;
end
%%
%% 基础粒子群优化算法
% fval,Mdl_best,gbest_t,Xfood
function [gBestScore,Mdl_best,cg_curve,gBest]=PSO_PSO(N,Max_iteration,lb,ub,dim,fobj,data_O,data_Z)

%PSO Infotmation
% if(max(size(ub)) == 1)
    UB=ub;
    LB=lb;
   ub = ub.*ones(1,dim);
   lb = lb.*ones(1,dim);  
% end

Vmax=6;
noP=N;
wMax=0.9;
wMin=0.6;
c1=2;
c2=2;

% Initializations
iter=Max_iteration;
vel=zeros(noP,dim);
pBestScore=zeros(noP);
pBest=zeros(noP,dim);
gBest=zeros(1,dim);
cg_curve=zeros(1,iter);
vel=zeros(N,dim);
pos=zeros(N,dim);

%Initialization
for i=1:size(pos,1) 
    for j=1:size(pos,2) 
        pos(i,j)=(ub(j)-lb(j))*rand()+lb(j);
        vel(i,j)=0.3*rand();
    end
end

for i=1:noP
    pBestScore(i)=inf;
end

% Initialize gBestScore for a minimization problem
 gBestScore=inf;
     
 [~,Mdl]=fobj(pos(1,:),data_O,data_Z);
  Mdl_best=Mdl; 
for l=1:iter 
    
    % Return back the particles that go beyond the boundaries of the search
    % space
     Flag4ub=pos(i,:)>ub;
     Flag4lb=pos(i,:)<lb;
     pos(i,:)=(pos(i,:).*(~(Flag4ub+Flag4lb)))+ub.*Flag4ub+lb.*Flag4lb;
%      disp(pos)
    for i=1:size(pos,1)     
        %Calculate objective function for each particle
        [fitness,Mdl]=fobj(pos(i,:),data_O,data_Z);

        if(pBestScore(i)>fitness)
            pBestScore(i)=fitness;
            pBest(i,:)=pos(i,:);
            Mdl_best=Mdl;
        end
        if(gBestScore>fitness)
            gBestScore=fitness;
            gBest=pos(i,:);
        end
    end

    %Update the W of PSO
    w=wMax-l*((wMax-wMin)/iter);
    %Update the Velocity and Position of particles
    for i=1:size(pos,1)
        for j=1:size(pos,2)       
            vel(i,j)=w*vel(i,j)+c1*rand()*(pBest(i,j)-pos(i,j))+c2*rand()*(gBest(j)-pos(i,j));
            
            if(vel(i,j)>Vmax)
                vel(i,j)=Vmax;
            end
            if(vel(i,j)<-Vmax)
                vel(i,j)=-Vmax;
            end            
            pos(i,j)=pos(i,j)+vel(i,j);
            if (pos(i,j)>UB(j))
                pos(i,j)=UB(j);
            elseif (pos(i,j)<LB(j))
                pos(i,j)=LB(j);
            end
        end
    end
    cg_curve(l)=gBestScore;
end

end
%% 模拟退火
%[gBestScore,Mdl_best,cg_curve,gBest]=PSO_PSO(N,Max_iteration,lb,ub,dim,fobj,data_O,data_Z)
function [Best_score,Mdl_best,curve,Best_pos]=SA_SA(N,Mmax,l,u,dim,fobj,data_O,data_Z)
%function [x0,f0]=sim_anl(f,x0,l,u,Mmax,TolFun)
% 输入: 
%        fobj = 适应度函数
%        x0 = 输入种群
%        l = 种群下边界
%        u = 种群上边界
%        Mmax = 最大温度
%        TolFun = 优化变化容忍度
%
% 输出: 
%        x0 = 输出优化后的种群
%        f0 = 输出优化后的种群的适应度值
TolFun = 10E-10;%模拟退火容忍度
x0 = (u-l).*rand(1,dim)+l;%随机初始化模拟退火;
% f = fobj;%适应度函数
x=x0;
% fx=feval(f,x);%计算适应度值
[fx,Mdl]=fobj(x0,data_O,data_Z);
Mdl_best=Mdl;
f0=fx;
count = 1;%用于记录收敛曲线标记
%模拟退火主要步骤
for m=1:Mmax
    T=m/Mmax; %温度
    mu=10^(T*1000);  
    %For each temperature we take 100 test points to simulate reach termal
    for k=0:N
        dx=mu_inv(2*rand(1,dim)-1,mu).*(u-l);
        if(isnan(dx))
            dx=0;
        end
%         disp(dx)
        x1=x+dx;
        %边界处理防止越界
        x1=(x1 < l).*l+(l <= x1).*(x1 <= u).*x1+(u < x1).*u;
%         disp(x1)
        %计算当前位置适应度值和适应度值偏差
        [fx1,Mdl]=fobj(x1,data_O,data_Z);df=fx1-fx;
        % 如果df<0则接受该解，如果大于0 则利用Metropolis准则进行判断是否接受       
        if (df < 0 || rand < exp(-T*df/(abs(fx)+eps)/TolFun))==1
            x=x1;fx=fx1;
            Mdl_best=Mdl;
        end        
        %判断当前解是否更优，更优则更新.       
        if fx1 < f0 ==1
            x0=x1;f0=fx1;
        end 
    end
     curve(count) = f0;
     count = count+1;
end
Best_pos = x0;
Best_score = f0;
end

function x=mu_inv(y,mu)
%模拟退火产生新位置偏差
x=(((1+mu).^abs(y)-1)/mu).*sign(y);
end
%%
function [fmin0,Mdl_best,ccurve,gbest]=WSO_WSO(whiteSharks,itemax,lb,ub,dim,fobj,data_O,data_Z)  
% Convergence curve
ccurve=zeros(1,itemax);

%%% Show the convergence curve
%     figure (1);
%     set(gcf,'color','w');
%     hold on
%     xlabel('Iteration','interpreter','latex','FontName','Times','fontsize',10)
%     ylabel('fitness value','interpreter','latex','FontName','Times','fontsize',10); 
%     grid;

% Start the WSO  Algorithm
% Generation of initial solutions
WSO_Positions=initialization(whiteSharks,dim,ub,lb);% Initial population

% initial velocity
v=0.0*WSO_Positions; 

% Evaluate the fitness of the initial population
fit=zeros(whiteSharks,1);

for i=1:whiteSharks
    [fit(i,1),Model1{1,i}]= fobj(WSO_Positions(i,:),data_O,data_Z);
%      fit(i,1)=fobj(WSO_Positions(i,:));
end

% Initalize the parameters of WSO
fitness=fit; % Initial fitness of the random positions of the WSO
 model2=Model1;
[fmin0,index]=min(fit);
Mdl_best=Model1{1,index};

wbest = WSO_Positions; % Best position initialization
gbest = WSO_Positions(index,:); % initial global position

% WSO Parameters
    fmax=0.75; %  Maximum frequency of the wavy motion
    fmin=0.07; %  Minimum frequency of the wavy motion   
    tau=4.11;  
       
    mu=2/abs(2-tau-sqrt(tau^2-4*tau));

    pmin=0.5;
    pmax=1.5;
    a0=6.250;  
    a1=100;
    a2=0.0005;
  % Start the iterative process of WSO 
for ite=1:itemax

    mv=1/(a0+exp((itemax/2.0-ite)/a1)); 
    s_s=abs((1-exp(-a2*ite/itemax))) ;
 
    p1=pmax+(pmax-pmin)*exp(-(4*ite/itemax)^2);
    p2=pmin+(pmax-pmin)*exp(-(4*ite/itemax)^2);
    
 % Update the speed of the white sharks in water  
     nu=floor((whiteSharks).*rand(1,whiteSharks))+1;

     for i=1:size(WSO_Positions,1)
           rmin=1; rmax=3.0;
          rr=rmin+rand()*(rmax-rmin);
          wr=abs(((2*rand()) - (1*rand()+rand()))/rr);       
          v(i,:)=  mu*v(i,:) +  wr *(wbest(nu(i),:)-WSO_Positions(i,:));
           %% or                

%          v(i,:)=  mu*(v(i,:)+ p1*(gbest-WSO_Positions(i,:))*rand+.... 
%                    + p2*(wbest(nu(i),:)-WSO_Positions(i,:))*rand);          
     end
 
 % Update the white shark position
     for i=1:size(WSO_Positions,1)
       
        f =fmin+(fmax-fmin)/(fmax+fmin);
         
        a=sign(WSO_Positions(i,:)-ub)>0;
        b=sign(WSO_Positions(i,:)-lb)<0;
         
        wo=xor(a,b);

        % locate the prey based on its sensing (sound, waves)
            if rand<mv
                WSO_Positions(i,:)=  WSO_Positions(i,:).*(~wo) + (ub.*a+lb.*b); % random allocation  
            else   
                WSO_Positions(i,:) = WSO_Positions(i,:)+ v(i,:)/f;  % based on the wavy motion
            end
    end 
    
    % Update the position of white sharks consides_sng fishing school 
for i=1:size(WSO_Positions,1)
        for j=1:size(WSO_Positions,2)
            if rand<s_s      
                
             Dist=abs(rand*(gbest(j)-1*WSO_Positions(i,j)));
             
                if(i==1)
                    WSO_Positions(i,j)=gbest(j)+rand*Dist*sign(rand-0.5);
                else    
                    WSO_Pos(i,j)= gbest(j)+rand*Dist*sign(rand-0.5);
                    WSO_Positions(i,j)=(WSO_Pos(i,j)+WSO_Positions(i-1,j))/2*rand;
                end   
            end
         
        end       
end
%     

% Update global, best and new positions
 
    for i=1:whiteSharks 
        % Handling boundary violations
           if WSO_Positions(i,:)>=lb & WSO_Positions(i,:)<=ub%         
            % Find the fitness
             [fit(i,1),Model1{1,i}]= fobj(WSO_Positions(i,:),data_O,data_Z);
%               fit(i)=fobj(WSO_Positions(i,:));    
              
             % Evaluate the fitness
            if fit(i)<fitness(i)
                 wbest(i,:) = WSO_Positions(i,:); % Update the best positions
                 fitness(i)=fit(i);   % Update the fitness
                 model2{1,i}=Model1{1,i};
            end
            
            % Finding out the best positions
            if (fitness(i)<fmin0)
               fmin0=fitness(i);
               gbest = wbest(index,:); % Update the global best positions
               Mdl_best=model2{1,index};
            end 
            
        end
    end

% Obtain the results
%   outmsg = ['Iteration# ', num2str(ite) , '  Fitness= ' , num2str(fmin0)];
%   disp(outmsg);
% 
%  ccurve(ite)=fmin0; % Best found value until iteration ite
% 
%  if ite>2
%         line([ite-1 ite], [ccurve(ite-1) ccurve(ite)],'Color','b'); 
%         title({'Convergence characteristic curve'},'interpreter','latex','FontName','Times','fontsize',12);
%         xlabel('Iteration');
%         ylabel('Best score obtained so far');
%         drawnow 
%  end 
  
end 
end
%%
function Positions=initialization(SearchAgents_no,dim,ub,lb)

Boundary_no= size(ub,2); % numnber of boundaries

% If the boundaries of all variables are equal and user enter a signle
% number for both ub and lb
if Boundary_no==1
    Positions=rand(SearchAgents_no,dim).*(ub-lb)+lb;
end

% If each variable has a different lb and ub
if Boundary_no>1
    for i=1:dim
        ub_i=ub(i);
        lb_i=lb(i);
        Positions(:,i)=rand(SearchAgents_no,1).*(ub_i-lb_i)+lb_i;
    end
end
end
%%
% Application of simple limits/bounds
function s = Bounds( s, Lb, Ub)
  % Apply the lower bound vector
  temp = s;
  I = temp < Lb;
  temp(I) = Lb(I);
  
  % Apply the upper bound vector 
  J = temp > Ub;
  temp(J) = Ub(J);
  % Update this new move 
  s = temp;
%   s=round(s);
end