clearvars; clearvars -global; clc; close all;

%% settings
addpath(genpath('Functions'));
addpath(genpath('./Libraries/'));

InitParam();


% fprintf('Preparing the training data\n');
% fprintf('***************************\n\n');
% PrepareTrainData();


fprintf('Preparing the test data\n');
fprintf('***********************\n\n');
PrepareTestData();