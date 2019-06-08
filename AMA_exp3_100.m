clear all
close all

num_d = 1100;
num_p = 100^2;
num_exp = 3;
NumItr = 1e4;
source_space = 1;
num_s = 25;

load('bags400_100_rescaled.mat')
% load('bags400_50_rescaled.mat')
f = f(:,1);

load('400vanSYM100_1e10_exp3.mat')
% load('400vanSYM2_50_1e10_exp3.mat')
I = reshape(I,25,[]);
I = I';

% figure(1)
% x_d = f/max(f);
% x_d = imadjust(x_d,[0 1],[0 1],0.6);    
% imagesc(reshape(x_d,100,100))
% axis image
% colormap(gray)        


load('H_1100_5m_sure.mat')
H_T = 0.1*H_T;
angl_T = photon_dist_T;


f = f(:);     
N = numel(f(:));  

Ht = H_T';
Z = max(sum(H_T,2));

r_m = angl_T.*exp(-H_T*f);
r_m = reshape(r_m,num_d,25);
rm = zeros(num_exp*num_d,1);
for e = 1:num_exp
    rm((e-1)*num_d+1:e*num_d) = sum(I(e,:).*r_m,2);
end
r = poissrnd(rm);

x_i = rand(N,1);
x_i = x_i/sum(x_i(:));

q = zeros(num_d*num_exp,25);
p = zeros(num_d*num_exp,25);

tic
x1 = x_i;
for i = 1:NumItr           
    r_m_temp = reshape(angl_T.*exp(-H_T*x_i),num_d,25);         
    for e = 1:num_exp
        q((e-1)*num_d+1:e*num_d,:) = I(e,:).*r_m_temp;
    end    
    p = (r.*q)./sum(q,2);

    p(isnan(p)) = 0;
    p(isinf(p)) = 0;    
    
    b_q = zeros(N,1);
    b_p = zeros(N,1);
    for e = 1:num_exp
       b_q = b_q + Ht*reshape(q((e-1)*num_d+1:e*num_d,:),[],1);
       b_p = b_p + Ht*reshape(p((e-1)*num_d+1:e*num_d,:),[],1);
    end    
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