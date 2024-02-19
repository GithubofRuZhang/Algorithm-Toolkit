function [responsivity,index]=gmm1(data,K)
        %GMM聚类
        X=data;
        [X_num,X_dim]=size(data);
        para_sigma_inv=zeros(X_dim, X_dim, K);
        N_pdf=zeros(X_num, K);  %单高斯分布的概率密度函数
        RegularizationValue=0.001;   %正则化系数，协方差矩阵求逆
        MaxIter=100;   %最大迭代次数
        TolFun=1e-8;   %终止条件
        gmm=fitgmdist(X, K, 'RegularizationValue', RegularizationValue, 'CovarianceType', 'diagonal', 'Start', 'plus', 'Options', statset('Display', 'final', 'MaxIter', MaxIter, 'TolFun', TolFun));
        NegativeLogLikelihood=gmm.NegativeLogLikelihood;
        NumIterations=gmm.NumIterations;  %迭代次数
        mu=gmm.mu;  %均值
        Sigma=gmm.Sigma;   %协方差矩阵
        ComponentProportion=gmm.ComponentProportion;  %混合比例
        for k=1:K
            sigma_inv=1./Sigma(:,:,k);  %sigma的逆矩阵,(X_dim, X_dim)的矩阵
            para_sigma_inv(:, :, k)=diag(sigma_inv);  %sigma^(-1)
        end
        for k=1:K
            coefficient=(2*pi)^(-X_dim/2)*sqrt(det(para_sigma_inv(:, :, k)));  %高斯分布的概率密度函数e左边的系数
            X_miu=X-repmat(mu(k,:), X_num, 1);  %X-miu: (X_num, X_dim)的矩阵
            exp_up=sum((X_miu*para_sigma_inv(:, :, k)).*X_miu,2);  %指数的幂，(X-miu)'*sigma^(-1)*(X-miu)
            N_pdf(:,k)=coefficient*exp(-0.5*exp_up);
        end
        responsivity=N_pdf.*repmat(ComponentProportion,X_num,1);  %响应度responsivity的分子，（X_num,K）的矩阵
        responsivity=responsivity./repmat(sum(responsivity,2),1,K);  %responsivity:在当前模型下第n个观测数据来自第k个分模型的概率，即分模型k对观测数据Xn的响应度
        %聚类
        [~,label]=max(responsivity,[],2);
        index=label';
        end