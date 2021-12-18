function InitParam()

global param;
global gamma;
global mu;

gamma = 2.2;
mu = 5000;

%%%%%%%%%%%%%%%%%% Parameters %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
param.trainingScenes = 'TrainingData/Training/';
param.trainingData = 'Result/Training/';

param.testScenes = 'TrainingData/Test/';
param.testData = 'Result/Test/';


param.cropSizeTraining = 10; % we crop the boundaries to avoid artifacts in the training
param.border = 6;












