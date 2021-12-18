function [contentNames, contentPaths, numContents] = GetFolderContent(folderPath, extension, onlyDir)

if(~exist('extension', 'var'))
    extension = [];
end

if(~exist('onlyDir', 'var'))
    onlyDir = false;
end

list = dir([folderPath, '/*', extension]);

fileNames = {list.name};
if (onlyDir)
    fileNames = fileNames([list.isdir]);
end

contentNames = setdiff(fileNames, {'.', '..'});
contentPaths = strcat(strcat(folderPath, '/'), contentNames);
numContents = length(contentNames);

