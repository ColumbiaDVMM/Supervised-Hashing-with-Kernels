function [a, cost] = OptProjectionFast(K, S, a0, cn)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% Projection Optimization Procedure of KSH
% Written by Wei Liu (wliu@ee.columbia.edu) 
% K(lXm) is the kernel matrix between l labeled samples and m anchors  
% S\in{1,-1}^{lXl} is the pairwise label matrix 
% a0(mX1) is the initial projection vector  
% cn is the iteration number 
% a(mX1) is the ouput optimized projection vector 
% cost is the output sequence of cost function values
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  Projection Optimization 
[n,m] = size(K);
cost = zeros(1,cn+2);
y = K*a0;
y = 2*(1+exp(-1*y)).^-1-1; %% to be changed
cost(1) = -y'*S*y;
cost(2) = cost(1); 
clear y;

a1 = a0; 
delta = zeros(1,cn+2);
delta(1) = 0;
delta(2) = 1;
beta = zeros(1,cn+1);
beta(1) = 1;

for t = 1:cn
    alpha = (delta(t)-1)/delta(t+1);
    v = a1+alpha*(a1-a0);           %% probe point
    y0 = K*v;
    y1 = 2*(1+exp(-1*y0)).^-1-1;    %% to be changed
    gv = -y1'*S*y1; 
    ty = (S*y1).*(ones(n,1)-y1.^2); %% to be changed
    dgv = -K'*ty;
    
    %% seek beta
    flag = 0;
    for j = 0:50
        b = 2^j*beta(t);
        z = v-dgv/b;
        y0 = K*z;
        y1 = 2*(1+exp(-1*y0)).^-1-1; %% to be changed
        gz = -y1'*S*y1;
        dif = z-v;
        gvz = gv+dgv'*dif+b*dif'*dif/2;
        clear dif;
        
        if gz <= gvz
            flag = 1;
            beta(t+1) = b;
            a0 = a1;
            a1 = z;
            cost(t+2) = gz; 
            break;
        end
    end
    
    if flag == 0
        t = t-1;
        break; 
    else
        delta(t+2) = ( 1+sqrt(1+4*delta(t+1)^2) )/2;
    end
       
    if abs(cost(t+2)-cost(t+1))/n <= 1e-2
        break;
    end 
end

a = a1;
clear K;
clear S;
clear a0;
clear a1;
clear v;
clear dgv;
clear z;
clear y0;
clear y1;
clear ty;
clear delta;
clear beta;

% figure;
% plot([1:t+2],cost(1:t+2)/n,'b');
cost = cost(1:t+2)/n;
