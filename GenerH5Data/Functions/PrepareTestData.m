function PrepareTestData()

global param;

sceneFolder = param.testScenes;
outputFolder = param.testData;

[sceneNames, scenePaths, numScenes] = GetFolderContent(sceneFolder, [], true);

MakeDir(outputFolder);

for i = 1 : numScenes
    
    count = fprintf('Started working on scene %d of %d', i, numScenes);
    
    %%% reading input data
    curExpo = ReadExpoTimes(scenePaths{i});
    [curImgsLDR, curLabel] = ReadImages(scenePaths{i});
    
    %%% processing data
    [inputs, label] = ComputeTestExamples(curImgsLDR, curExpo, curLabel);
    
    %%% writing data
%     label=[];
%     inputs = ((curImgsLDR{1}));
%     inputs = cat(3, inputs, (curImgsLDR{2}));
%     inputs = cat(3, inputs, (curImgsLDR{3}));
    WriteTestExamples(inputs, label, [outputFolder, '/', sceneNames{i}, '.h5']);
%     WriteTestExamples(inputs, label, [outputFolder, '/', 'TrainSequence', num2str(i) '.h5']);
    
    fprintf(repmat('\b', [1, count]));
end

fprintf('Done\n\n');