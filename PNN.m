function [I_out, I_MS] = PNN(I_MS_LR, I_PAN, param)
%I_MS_HR = PNN(I_MS_LR, I_PAN, PNN_model);
% I_out = PNN(I_MS_LR, I_PAN, param)
%
%PNN (Pansharpening by Neural Networks) algorithm is
%   described in "Pansharpening by Convolutional Neural Networks", 
%   written by  G. Masi, D. Cozzolino, L. Verdoliva and G. Scarpa, 
%   Remote Sensing, 2016.
%   Please refer to this paper for a more detailed description of
%   the algorithm.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% Copyright (c) 2016 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
% All rights reserved.
% This work should only be used for nonprofit purposes.
% 
% By downloading and/or using any of these files, you implicitly agree to all the
% terms of the license, as specified in the document LICENSE.txt
% (included in this package) and online at
% http://www.grip.unina.it/download/LICENSE_OPEN.txt
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Analyze Input Arguments
if ischar(param), param = load(param); end;
if not(isfield(param,'inputType')),  param.inputType = 'MS_PAN'; end;
mav_value = 2^param.L;
NDxI_LR = [];
I_MS_LR = double(I_MS_LR);
I_PAN   = double(I_PAN);
if isfield(param,'net'),
    net = param.net;
else
    net = dagnn.DagNN.loadobj(param.model_net);
end

% Compute Radiometric Indexes %计算辐射指数
if isequal(param.inputType,'MS_PAN_NDxI'),
    if size(I_MS_LR,3) == 8,
        NDxI_LR = cat(3,...
              (I_MS_LR(:,:,5)-I_MS_LR(:,:,8))./(I_MS_LR(:,:,5)+I_MS_LR(:,:,8)), ...
              (I_MS_LR(:,:,1)-I_MS_LR(:,:,8))./(I_MS_LR(:,:,1)+I_MS_LR(:,:,8)), ...
              (I_MS_LR(:,:,3)-I_MS_LR(:,:,4))./(I_MS_LR(:,:,3)+I_MS_LR(:,:,4)), ...
              (I_MS_LR(:,:,6)-I_MS_LR(:,:,1))./(I_MS_LR(:,:,6)+I_MS_LR(:,:,1)) );
    else
        NDxI_LR = cat(3,...
              (I_MS_LR(:,:,4)-I_MS_LR(:,:,3))./(I_MS_LR(:,:,4)+I_MS_LR(:,:,3)), ...
              (I_MS_LR(:,:,2)-I_MS_LR(:,:,4))./(I_MS_LR(:,:,2)+I_MS_LR(:,:,4)) );
    end;
end;
% Input Preparation  %输入前准备
if isequal(param.typeInterp,'interp23tap'),
    I_MS = interp23tap(I_MS_LR, param.ratio);
    if not(isempty(NDxI_LR)), NDxI = interp23tap(NDxI_LR,param.ratio); end;
elseif isequal(param.typeInterp,'cubic'),
    I_MS = imresize(I_MS_LR,size(I_PAN),'bicubic');
    if not(isempty(NDxI_LR)), NDxI = imresize(NDxI_LR,size(I_PAN),'bicubic'); end;
else
    error('Interpolation not supported');
end;
if isequal(param.inputType,'MS'),
    I_in = single(I_MS)/mav_value;
elseif isequal(param.inputType,'MS_PAN'),
    I_in = single(cat(3,I_MS,I_PAN))/mav_value;
elseif isequal(param.inputType,'MS_PAN_NDxI'),
    I_in = single(cat(3,I_MS,I_PAN))/mav_value;
    I_in = single(cat(3,I_in,single(NDxI)));
else
   error('Configuration not supported');
end;
I_in = padarray(I_in, [param.padSize,param.padSize]/2, 'replicate','both');

% Pansharpening  
net.eval({'data',I_in});
I_out = net.vars( net.getVarIndex(net.getOutputs{1}) ).value;
I_out = I_out(:,:,1:size(I_MS,3))*mav_value;


