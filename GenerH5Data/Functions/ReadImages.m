function [input, label] = ReadImages(scenePath)

[~, imgPaths, numImages] = GetFolderContent(scenePath, 'tif');

input = cell(1, numImages);
for k = 1 : numImages
    input{k} = im2single(imread(imgPaths{k}));
    input{k} = Clamp(input{k}, 0, 1);
end

refFile = [scenePath, '/HDRImg.hdr'];
if (exist(refFile, 'file'))
    label = hdrread(refFile);
else
    label = [];
end