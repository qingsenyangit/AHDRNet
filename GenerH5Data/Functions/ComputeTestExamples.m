function [inputs, label] = ComputeTestExamples(curImgsLDR, curExpo, curLabel)

global param;

cropSize = param.cropSizeTraining;
border = param.border;

%%% prepare input features
[inputs, label] = PrepareInputFeatures(curImgsLDR, curExpo, curLabel, true);

% %%% crop boundaries
% inputs = CropImg(inputs, cropSize-border);
% label = CropImg(label, cropSize-border);
