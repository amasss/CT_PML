clear all
close all

num_d = 1100;
num_p = 100^2;
NumItr = 1e4;
source_space = 1;
num_s = 25;

I0 = 1e10;

load('bags400_100_rescaled.mat')
f = f(:,1);

% figure(1)
% x_d = f/max(f);
% x_d = imadjust(x_d,[0 1],[0 1],0.6);    
% imagesc(reshape(x_d,50,50))
% axis image
% colormap(gray)        


load('H_1100_5m_sure.mat')
H_T = 0.1*H_T;
angl_T = photon_dist_T;


f = f(:);     
N = numel(f(:));  

indx_nz = find(sum(H_T,2) > 0);
H = H_T(indx_nz,:);
angl = angl_T(indx_nz);
clear angl_T H_T 

Ht = H';
r_m = (I0/num_s).*angl.*exp(-H*f);        
Z = max(sum(H,2)); 
r = poissrnd(r_m);
b = Ht*r;

x_i = rand(N,1);
x_i = x_i/sum(x_i(:));

tic
x1 = x_i;
for i = 1:NumItr           
    proj = H*x_i;                                                  
    q = (I0/num_s)*angl.*exp(-proj);                                           
    b_E = H'*(q);                                                 
    x_i = x_i + log(b_E./b)./Z;                                 
    x_i((x_i<0)) = 0;                                            
    convg = norm(x_i - x1)/norm(x1);
    disp([num2str(i) '     : ' num2str(convg)])    
    x1 = x_i;
    if mod(i,100) == 0        
        rmse = sqrt(sum((x_i - f).^2)/N);
        nrmse = rmse/(max(f) - min(f));
        x_d = x_i/max(x_i);
        x_d = imadjust(x_d,[0 1],[0 1],0.6);    
        imagesc(reshape(x_d,100,100))
        axis image
        colormap(gray)        
        disp(['nrmse = ' num2str(nrmse)])
    end
end
toc