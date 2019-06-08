close all
clear all

maxNumCompThreads('automatic')
% maxNumCompThreads(2)
% maxNumCompThreads(2)

% id = getenv('SGE_TASK_ID');
% id = num2str(id);
% id = str2num(id);
% disp(num2str(id));
% disp('.')
% disp('.')

id = 4;

load('stream.mat')
RandStream.setGlobalStream(s{id});
clear s


I0_vec = [1e7 1e8 1e9 1e10];
num_ph_vec = [{'1e7'},{'1e8'},{'1e9'},{'1e10'}];
lambda_vec = [1e5 1e5 1e5 1e6];
lambda_txt = [{'1e5'},{'1e5'},{'1e5'},{'1e6'}];


num_ph = num_ph_vec{id};
I0 = I0_vec(id);
lambda_txt = lambda_txt{id};
lambda = lambda_vec(id);



num_d = 1100;
num_p = 100^2;
NumItr = 1e2;
NumGrad = 100;
num_expr = '3';
source_space = 1;
source_indx = (1:source_space:25);
num_s = numel(1:source_space:25);
delta = 0.001;

% load('bags400_50_test_rescaled.mat')
% load('bag50_train2_journal.mat')
load('bags400_100_rescaled.mat')
f = f(:,1);
f = f(:);     
N = numel(f(:));  

load('H_1100_5m_sure.mat')
H_T = 0.1*H_T;
angl_T = photon_dist_T;
clear photon_dist_T

load(['400vanSYM100_' num_ph '_exp3.mat']); %400vanSYM_1e7_exp1
Imin = I;

HH = sparse(str2double(num_expr)*size(H_T,1),num_p);
for i = 1:num_s
    HH((i-1)*str2double(num_expr)*num_d+1:i*str2double(num_expr)*num_d,:) = repmat(H_T((i-1)*num_d+1:i*num_d,:),str2double(num_expr),1);
end


I1 = repmat(Imin(1:num_s),num_d,1);
I2 = repmat(Imin(num_s+1:2*num_s),num_d,1);
I3 = repmat(Imin(2*num_s+1:3*num_s),num_d,1);


r1_m = reshape((I1(:)).*angl_T.*exp(-H_T*f(:)),num_d,num_s);        
r2_m = reshape((I2(:)).*angl_T.*exp(-H_T*f(:)),num_d,num_s);        
r3_m = reshape((I3(:)).*angl_T.*exp(-H_T*f(:)),num_d,num_s);        

rm = sum([r1_m;r2_m;r3_m],2);     

down = 1;
alpha = 1/down;

Z = max(sum(H_T,2)); 

y = poissrnd(rm);

% b = Ht*r;


x_i = rand(N,1);
x_i = x_i/sum(x_i);
% x_i = zeros(N,1);
convg = [];

tic
for i = 1:NumItr
    disp(['+++++++ ' num2str(i)])
    q1 = (I1(:)).*angl_T.*exp(-H_T*x_i);                                                
    q2 = (I2(:)).*angl_T.*exp(-H_T*x_i);
    q3 = (I3(:)).*angl_T.*exp(-H_T*x_i);
    
    Q1 = reshape(q1,num_d,num_s);
    Q2 = reshape(q2,num_d,num_s);   
    Q3 = reshape(q3,num_d,num_s);   
    
    Q = [Q1;Q2;Q3];       
%     q = (I0/num_s).*angl_T.*exp(-H_T*x_i);                                                   
%     Q = reshape(q,num_d,num_s); %% Q is q_est
    % Computing p(y,j)
    q_sum = sum(Q,2);
    y_q_ratio = y./q_sum;     
    y_q_ratio = repmat(y_q_ratio,1,num_s);           
    p = Q.*(y_q_ratio);
    p(Q == 0) = 0;
    p(isnan(p)) = 0;
    p(isinf(p)) = 0;     
    y_q_ratio(isnan(y_q_ratio)) = 0;
    y_q_ratio(isinf(y_q_ratio)) = 0;

    
    Q_repmat = bsxfun(@times, HH, Q(:));
    p_repmat = bsxfun(@times, Q_repmat, y_q_ratio(:));    
    grad_1_1 = sum(p_repmat)';
    
    %% newton
    for j = 1:NumGrad
        if j == 1
            minn = 1e20;
            x_min = zeros(N,1); 
            if i == 1
                minn = 1e20;
                x_min = zeros(N,1); 
            end
%               x_j = zeros(N,1);  
              x_j = rand(N,1);                 
              x_j = x_j/max(x_j);                   
        end

        rj = reshape(H_T*x_j,num_d,num_s);                                                
        rj_exp = reshape(H_T*exp(-Z*(x_j - x_i)),num_d,num_s)/Z;
        yd = sum(sum(p.*repmat(rj,str2double(num_expr),1) + Q.*repmat(rj_exp,str2double(num_expr),1)));

%         Q_repmat = bsxfun(@times, HH, Q(:));
%         p_repmat = bsxfun(@times, Q_repmat, y_q_ratio(:));
        
%         grad_1_1 = sum(repmat(p(:),1,num_p).*HH)';
%         grad_1_2 = (sum(repmat(Q(:),1,num_p).*HH)').*exp(-Z*(x_j - x_i));
%         grad_1_1 = sum(p_repmat)';
        grad_1_2 = (sum(Q_repmat)').*exp(-Z*(x_j - x_i));
        grad_1 = grad_1_1 - grad_1_2;

%         Hess_1 = (sum(repmat(Q(:),1,num_p).*HH)').*exp(-Z*(x_j - x_i))*Z;
        Hess_1 = grad_1_2*Z;
        [yr,grad_2,Hess_2] = R_H_s(x_j,x_i,delta);        
%         disp(num2str((yd+lambda*yr)));

        if j == 1
            y_1 = yd+lambda*yr;
        elseif j == 2
            y_2 = yd+lambda*yr;
        else
            y_1 = y_2;
            y_2 = yd+lambda*yr;
        end        
        
        
        if (yd + lambda*yr) < minn
            minn = (yd + lambda*yr);
            x_min = x_j;
        end
        grad_2 = lambda*grad_2;        
        grad = (grad_1 + grad_2);
%         grad = grad/norm(grad);
        Hess_inv = diag(1./(Hess_1 + lambda*Hess_2));       
        x_temp = x_j;
        dir = Hess_inv*grad;
%         dir = alpha*down*dir/norm(dir);
        x_j = x_j - dir;
        x_j(x_j < 0) = 0;
%         indx = abs(x_j - x_temp)./abs(x_temp);
%         indx(isnan(indx)) = 0;
%         indx(isinf(indx)) = 0;
        if j > 1
            if abs(y_2 - y_1) < 1e-5 && ( j > 2)
                convg = [convg norm(x_i - x_j,2)/norm(x_i,2)];
                x_i = x_j;
                break;
            end
        end
        if j == NumGrad
            convg = [convg norm(x_i - x_j,2)/norm(x_i,2)];
            x_i = x_j;           
        end        
    end    

    
    if any(isnan(x_i))
        error('Not A Number !')
    end
    if i >= 2
        disp([' convg = ' num2str(convg(end))]);        
    end
    if i >= 200
        if convg(end) >= (0.1*norm(x_i,2))
            error('Diverge!')
        end
    end
    if i >= 1000
        if convg(end) >= 0.001
            error('Diverge!')
        end
    end
%     if mod(i,100) == 0    
%         rmse = sqrt(sum((x_i - f).^2)/num_p);
%         nrmse = rmse/(max(f) - min(f));
%         cd('./fig_journal')
%         %cd('./fig_patent')
%         txt = ['50MUX3_train2_' num_ph '_' lambda_txt '_' num2str(id) '.mat'];
%         save(txt,'nrmse','rmse','x_i','convg')                
% 	if mod(i,1e4) == 0
% 	    tt = toc;
% 	    save(txt,'nrmse','rmse','x_i','tt')
% 	end
%         cd('../')
%     end    
end
toc

% close all
figure(2)
x_d = x_i/max(x_i);
x_d = imadjust(x_d,[0 1],[0 1],0.4);    
imagesc(reshape(x_d,100,100))
axis image
colormap(gray)
rmse = sqrt(sum((x_i - f).^2)/N);
nrmse = rmse/(max(f) - min(f))
% cd('C:\Users\Ahmad\Desktop')
% saveas(gcf,'test.jpeg')
% cd('C:\Users\Ahmad\Desktop\Joint Det_Est')