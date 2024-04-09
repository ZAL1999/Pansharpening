clear all; close all; clc;
%% Setup  %%MatConvNet是一个实现用于计算机视觉应用的卷积神经网络（CNN）的MATLAB工具箱
% Set Matconvnet Path
matConvnetPath = 'D:\matlab\bin\matconvnet\matconvnet-1.0-beta25';
run(fullfile(matConvnetPath,'matlab\vl_setupnn'));


%% Load Input
% Set the sensor
% sensor = 'GeoEye1';
sensor = 'WV2';
%sensor = 'IKONOS';

% load sensor data
switch sensor
    case 'GeoEye1'
        PNN_model = './networks/GeoEye1/PNN_07_09_48/PNN_model_iter1120000.mat';
        inputImage = load('./imgs/imgGeoEye1.mat');
    case 'IKONOS'
        PNN_model = './networks/IKONOS/PNN_07_09_64/PNN_model_iter1120000.mat';
        inputImage = load('./imgs/imgIKONOS.mat');
    case 'WV2'
        PNN_model = './networks/WV2/PNN_13_09_56/PNN_model_iter1120000.mat';
        inputImage = load('./imgs/Test(HxWxC)_wv2_data4.mat');
    otherwise
        error('sensor not supported'),
end
I_MS_LR = inputImage.I_MS;
I_PAN   = inputImage.I_PAN;
RGB_indexes = inputImage.RGB_indexes;
        
%% Pansharpensing
timestamp = tic();
I_MS_HR = PNN(I_MS_LR, I_PAN, PNN_model);
time_sec = toc(timestamp);


%% Visualization
figure();
subplot(1,3,1);
th_PAN = image_quantile(I_PAN, [0.01 0.99]);
imshow( image_stretch(I_PAN,th_PAN)); title('Panchromatic');

subplot(1,3,2);
th_MSrgb = image_quantile(I_MS_LR(:,:,RGB_indexes), [0.01 0.99]);
imshow(image_stretch(I_MS_LR(:,:,RGB_indexes),th_MSrgb)); title('Multispectral low-resolution');

subplot(1,3,3);
imshow(image_stretch(I_MS_HR(:,:,RGB_indexes),th_MSrgb)); title('Multispectral high-resolution');

% load DEIMOS
inputImage1 = load('./PNN_NIHS/PNN_GeoEye1/I_MS_HR_CM0043.mat');
inputImage2 = load('./PNN_NIHS/PNN_GeoEye1/I_PAN_CM0043.mat');

LRM = inputImage1.I_MS_HR;
HRP = inputImage2.I_PAN;


% LRM = im2double( imread('LR_11_ple.tif'));
% HRP = im2double( imread('pan_11.tif'));
% Warning! 
% This data is provided by the 2016 IEEE GRSS Data Fusion Contest. In any 
% scientific publication using the data, the data shall be identified as 
% 揼rss_dfc_2016?and shall be referenced as follows: 揫REF. NO.] 2016 IEEE 
% GRSS Data Fusion Contest. 
% Online: http://www.grss-ieee.org/community/technical-committees/data-fusion?

% Any scientific publication using the data shall include a section 
% 揂cknowledgement? This section shall include the following sentence: 
% 揟he authors would like to thank Deimos Imaging for acquiring and 
% providing the data used in this study, and the IEEE GRSS Image Analysis 
% and Data Fusion Technical Committee.?
rho = 4;
SB  = [3 2 1];
% for i = 1:size(LRM, 3)
%     LRMup(:,:,i)  = imresize(LRM(:,:,i), [size(HRP,1), size(HRP,2)], 'bicubic'); % Upsampled LRM data
% end
LRMup = LRM;
imshow(HRP); 
% title('Panchromatic data')

th_MSrgb = image_quantile(LRMup(:,:,SB), [0.01 0.99]);
imshow(image_stretch(LRMup(:,:,SB),th_MSrgb)); 
title(['Upsampled multispectral data, bands: ', num2str(SB)]);
% imshow(LRMup(:,:, SB));
% figure;imshow(LRMup(:,:, SB)) ; title(['Upsampled multispectral data, bands: ', num2str(SB)])

% Nonlinear IHS ==========================================================
% specify the parameter settings
P       = 5;      % patch size for partioning input images
q       = 3;      % overlap between adjacent patches

eta     = 1;      % balanced parameter of global synthesis[see equ.(17)]
maxIter = 10;     % the number of iterations in global synthesis procedure
mu      = maxIter^-1; % step size of the gradient descent[see equ.(17)]
gamma   = 10^-9;  % gamma controls the magnitude on the edges[see eq.(5)]
eps     = 10^-10; % eps enforces a nonzero denominator [see eq.(5)]

[NIHS, Iuphat, I, ~] = NonlinearIHS(HRP, LRMup, P, q, rho, maxIter, mu, eta, gamma, eps);
% [NIHS, Iuphat, I, ~] = NonlinearIHS(HRP, LRMup, P, q, rho, maxIter, mu, eta, gamma, eps);
figure;imshow(Iuphat); 
title('Iuphat');
figure;imshow(I); title('I');
th_MSrgb = image_quantile(LRMup(:,:,SB), [0.01 0.99]);
imshow(image_stretch(NIHS(:,:,SB),th_MSrgb));
title('NIHS');
