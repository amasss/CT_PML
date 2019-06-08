close all
clear all

maxNumCompThreads('automatic')

% maxNumCompThreads(2)
% maxNumCompThreads(2)

id = 4;

load('stream.mat')
RandStream.setGlobalStream(s{id});
clear s

I0_vec = [1e7 1e8 1e9 1e10]/25;
num_ph_vec = [{'1e7'},{'1e8'},{'1e9'},{'1e10'}];
lambda_vec = [1e5 1e6 1e6 1e7]/25;
lambda_txt = [{'1e5'},{'1e6'},{'1e6'},{'1e7'}];

num_ph = num_ph_vec{id};
I0 = I0_vec(id);
lambda_txt = lambda_txt{id};
lambda = lambda_vec(id);

num_d = 1100;
num_p = 100^2;
NumItr = 1e2;
NumGrad = 100;
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

indx_nz = find(sum(H_T,2) > 0);
H = H_T(indx_nz,:);
angl = angl_T(indx_nz);
clear angl_T H_T photon_dist_T

Ht = H';
r_m = (I0/num_s).*angl.*exp(-H*f);        
down = 1;
alpha = 1/down;
Z = max(sum(H,2)); 
r = poissrnd(r_m);

b = Ht*r;

x_i = rand(num_p,1);
x_i = x_i/sum(x_i(:));

convg = [];
tic;
for i = 1:NumItr
    disp(['+++++++ ' num2str(i)])
    q_est = (I0/num_s).*angl.*exp(-H*x_i);        
    b_est = Ht*q_est;
%     y1 = b.*x_i + (b_est/Z).*exp(-Z*(x_));
    %% newton
    for j = 1:NumGrad
        if j == 1
            minn = 1e20;
            x_min = zeros(num_p,1); 
            if i == 1
                minn = 1e20;
                x_min = zeros(num_p,1); 
            end
              x_j = zeros(num_p,1);  
%               x_j = rand(N,1);                 
%               x_j = x_i;
%               x_j = x_j/max(x_j);                   
        end        
        yd = sum(b.*x_j + (b_est/Z).*exp(-Z*(x_j - x_i)));
        grad_1 = b - b_est.*exp(-Z*(x_j - x_i));         
        Hess_1 = b_est.*exp(-Z*(x_j - x_i))*Z;
%         [yr,grad_2] = R_q(x_j,x_i);
%         Hess_2 = ones(N,1)*4*(0.5*4 + 2/sqrt(2));        
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
        if convg(end) >= 0.1
            error('Diverge!')
        end
    end
%     if mod(i,100) == 0        
%         rmse = sqrt(sum((x_i - f).^2)/num_p);
%         nrmse = rmse/(max(f) - min(f));
%         x_d = x_i/max(x_i);
%         x_d = imadjust(x_d,[0 1],[0 1],0.6);    
%         imagesc(reshape(x_d,100,100))
%         axis image
%         colormap(gray)        
%         cd('./fig_journal')
%         txt = ['50conv2_train2_' num_ph '_' lambda_txt '_' num2str(id) '.mat'];
%         save(txt,'nrmse','rmse','x_i')                
% 	if mod(i,1e4) == 0
% 	    tt = toc;
% 	    save(txt,'nrmse','rmse','x_i','tt')
% 	end 
%         cd('../')
%     end
end

toc

close all
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