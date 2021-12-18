clear; clc;

%%% Parameters ---------------
newWidth = 1500;
newHeight = 1000;
sceneName = 'LadySitting';
sensorAlignment = 'rggb';
%%% --------------------------

gamma = 2.2;
inputSceneFolder = 'Scenes';

sceneFolder = sprintf('%s\\%s', inputSceneFolder, sceneName);

if (~exist(sceneFolder, 'dir'))
    error('Scene folder does not exist');
end

listOfFiles = dir(sprintf('%s\\*.pgm', sceneFolder));
numImages = size(listOfFiles, 1);
inputLDRs = cell(1, numImages);

for i = 1 : numImages
    Path = sprintf('%s\\%s\\%s', inputSceneFolder, sceneName, listOfFiles(i).name);
    inputLDRs{i} = imread(Path);
    
    % Demosaicing the input
    inputLDRs{i} = double(demosaic(inputLDRs{i}, sensorAlignment));
    
    % Gamma correction
    inputLDRs{i} = (inputLDRs{i}/2^16).^(1/gamma);
    inputLDRs{i} = uint16(inputLDRs{i}*2^16);
    
    % Resizing
    inputLDRs{i} = imresize(inputLDRs{i}, [newHeight  newWidth], 'bicubic');
    
    % Saving the image and deleting the pgm file
    imwrite(inputLDRs{i}, sprintf('%s\\%s\\%s.tif', inputSceneFolder, sceneName, listOfFiles(i).name(1:end-4)));
end