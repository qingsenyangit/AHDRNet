function expoTimes = ReadExpoTimes(scenePath)

[~, expPaths, ~] = GetFolderContent(scenePath, '.txt');

fid = fopen(expPaths{1});
expoTimes = 2.^cell2mat(textscan(fid, '%f'));
fclose(fid);