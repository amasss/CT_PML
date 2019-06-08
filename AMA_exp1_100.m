clear all
close all

num_d = 1100;
num_p = 100^2;
num_exp = 1;
NumItr = 1e3;
source_space = 1;
num_s = 25;

load('bags400_100_rescaled.mat')
% load('bags400_50_rescaled.mat')
f = f(:,1);

load('400vanSYM100_1e10_exp1.mat')
% load('400vanSYM2_50_1e8_exp1.mat')

figure(1)
x_d = f/max(f);
x_d = imadjust(x_d,[0 1],[0 1],0.6);    
imagesc(reshape(x_d,sqrt(num_p),sqrt(num_p)))
axis image
colormap(gray)        


load('H_1100_5m_sure.mat')
H_T = 0.1*H_T;
angl_T = photon_dist_T;


f = f(:);     
N = numel(f(:));  

Ht = H_T';
Z = max(sum(H_T,2));

r_m = angl_T.*exp(-H_T*f);        
rm = sum(I.*reshape(r_m,num_d,25*num_exp),2);
r = poissrnd(rm);

x_i = rand(N,1);
x_i = x_i/sum(x_i(:));

tic
x1 = x_i;
for i = 1:NumItr           
    r_m_temp = angl_T.*exp(-H_T*x_i);                             
    q = I.*reshape(r_m_temp,num_d,25*num_exp);                                           
    p = (r.*q)./sum(q,2);

    p(isnan(p)) = 0;
    p(isinf(p)) = 0;    
    
    b_q = Ht*(q(:));
    b_p = Ht*(p(:));
    
    L = log(b_q./b_p);
    L(isnan(L)) = 0;
    L(isinf(L)) = 0;    
    x_i = x_i + L/Z;                                 
    x_i((x_i<0)) = 0;    
    
    convg = norm(x_i - x1)/norm(x1);  
    disp([num2str(i) '     : ' num2str(convg)]) 
    x1 = x_i;
%     if mod(i,1000) == 0                
%         disp([num2str(i) '     : ' num2str(convg)]) 
%         rmse = sqrt(sum((x_i - f).^2)/N);
%         nrmse = rmse/(max(f) - min(f));
%         x_d = x_i/max(x_i);
%         x_d = imadjust(x_d,[0 1],[0 1],0.6);    
%         imagesc(reshape(x_d,100,100))
%         axis image
%         colormap(gray)        
%         disp(['nrmse = ' num2str(nrmse)])
%     end
end
toc

rmse = sqrt(sum((x_i - f).^2)/N);
nrmse = rmse/(max(f) - min(f));
x_d = x_i/max(x_i);
x_d = imadjust(x_d,[0 1],[0 1],0.6); 
figure(2)
imagesc(reshape(x_d,sqrt(num_p),sqrt(num_p)))
axis image
colormap(gray)        
disp(['nrmse = ' num2str(nrmse)])