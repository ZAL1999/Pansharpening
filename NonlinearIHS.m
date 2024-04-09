function [F, Iuphat, I, HRPhis] = NonlinearIHS(HRP, LRMup, P, q, rho,...
 maxIter, mu, eta, gamma, eps)
	
% Input
%  HRP       High Resolution Panchromatic data
%  LRMup     Upsampled Low Resolution Multispectral data
%  p         patch size for partioning input images
%  O         overlap between adjacent patches
%  rho       the ratio of the spatial resolutions between the LRM and HRP
%  eta       the balanced parameter of global synthesis[see equ.(17)]
%  maxIter   the number of iterations in global synthesis procedure
%  mu        step size of the gradient descent[see equ.(17)]
%  gamma     this parameter controls the magnitude on the edges[see eq.(5)]
%  eps       this parameter enforces a nonzero denominator [see eq.(5)]

% Outputs
%	 F            Fused/pan-sharpened bands
%	 Iuphat       Upsampled Intensity component
%	 I            Intensity component
%    HRPhis       histogram-matched HRP data

% Abbreviations
%   HR  High Resolution
%   LR  Low Resolution

% Description
%   Nonlinear IHS is a two step pansharpening algorithm to estimate the
%   pan-sharpened bands
%
% Aug. 2, 2016. Morteza Ghahremani 
% Image Processing and Data Analysis Laboratory
% Tarbiat Modares University
%
% References
% Morteza Ghahremani and Hassan Ghassemian. Nonlinear IHS: A Promising 
% Method for Pan-Sharpening. IEEE Geoscience and Remote Sensing Letter
%
% For any questions, please do not hesitate and email me by 
% morteza.ghahremani.b@gmail.com
%% 1- Preprocess
warning off
LRMup = double(LRMup);
HRP   = double(HRP);
[rHR, cHR, L] = size(LRMup);

% Normalize input images
for i = 1:L
    LRMCoeffs(i) = max(max(abs(LRMup(:,:,i))));
    LRMup(:,:,i) = LRMup(:,:,i)/LRMCoeffs(i);
    LRM(:,:,i)   = imresize(LRMup(:,:,i), [size(HRP,1)/rho, size(HRP,2)/rho], 'bicubic'); % Low Resolution Multispectral data
end

HRPCoeffs = max(max(abs(HRP)));
HRP  = HRP/HRPCoeffs;
LRP  = imresize(HRP, [size(HRP,1)/rho, size(HRP,2)/rho], 'bicubic'); % Low Resolution Panchromatic data

%==========================================================================
% 2- Local Synthesis  (phase I)
prompt = 'Please choose the type of Local Synthesis{(1).m function,(2)mex file}: 1 or 2?';
result = input(prompt);

disp('Please wait...');

if result == 1
    Type = '.m function';
    tic
    [Iup0, I, HRPr] = LocalSynthesis(HRP, LRP, LRMup, LRM, P, q, rho);
    t = toc ;
else
    Type = 'mex file';
    LRMupre = reshape(LRMup, [rHR*cHR, L]);
    LRMre   = reshape(LRM  , [rHR*cHR/rho^2, L]);
    tic
    [Iup0, I, HRPr] = LocalSynthesis_mex(HRP, LRP, LRMupre, LRMre, P, q, rho);
    t = toc;
    clear  LRMupre LRMre
end

disp(['Elapsed time for the local synthesis of "', Type, '" is ',num2str(t),' sec.'])
%==========================================================================
% 3- Global Synthesis (phase II)
tic
Iuphat = GlobalSynthesis(Iup0, I , maxIter, mu, eta);
t = toc;

disp(['Elapsed time for the global synthesis is ',num2str(t),' sec.'])
%% 4- Pan-sharpening
% 4-1. Histogram Matching
HRPhis = (HRPr-mean(HRPr(:)))*std(Iuphat(:))/std(HRPr(:)) + mean(Iuphat(:));


% 4-2. Edge Detector 
E = expEdge(HRPhis, gamma, eps);

clear F
for i = 1:L
    Details = double(E.*(HRPhis-Iuphat));
    F(:,:,i) = (LRMup(:,:,i)+ Details)* LRMCoeffs(i);
end
warning on
end
%%%%%%%%%%%%%%%
function out = expEdge(HRP, gamma, eps)
%% See equation (5)
[Gx,Gy] = gradient(HRP);
u =((Gx.*Gx)+(Gy.*Gy)).^(1/2);

[m,n]= size(u); 
out = exp(-((gamma*(ones(m,n)))./((abs(u).*abs(u).*abs(u).*abs(u))+ eps)));
end