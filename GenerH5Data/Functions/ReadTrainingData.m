function [inputs, reference] = ReadTrainingData(fileName, isTraining, it)


global param;
batchSize = param.batchSize;
border = param.border;
useGPU = param.useGPU;

if (~exist('isTraining', 'var') || isempty(isTraining))
    isTraining = true;
end

fileInfo = h5info(fileName);
numItems = length(fileInfo.Datasets);
maxNumPatches = fileInfo.Datasets(1).Dataspace.Size(end);
numImages = floor(maxNumPatches / batchSize) * batchSize;

if (isTraining)
    startInd = mod((it-1) * batchSize, numImages) + 1;
else
    startInd = 1;
    batchSize = 1;
end

reference = []; inputs = [];

for i = 1 : numItems
    
    dataName = fileInfo.Datasets(i).Name;
    
    switch dataName
        
        case 'GT'
            s = fileInfo.Datasets(i).Dataspace.Size;
            reference = h5read(fileName, '/GT', [1, 1, 1, startInd], [s(1), s(2), s(3), batchSize]);
            reference = single(CropImg(reference, border));
            reference = RangeCompressor(reference);
            if (useGPU)
                reference = gpuArray(reference);
            end
            
        case 'IN'
            s = fileInfo.Datasets(i).Dataspace.Size;
            inputs = h5read(fileName, '/IN', [1, 1, 1, startInd], [s(1), s(2), s(3), batchSize]);
            if (useGPU)
                inputs = gpuArray(inputs);
            end
            
    end
end


    
