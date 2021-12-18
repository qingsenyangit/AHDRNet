function endInd = SaveHDF(fileName, datasetName, input, inDims, startLoc, chunkSize, createFlag, arraySize)

if (~exist('arraySize', 'var') || isempty(arraySize))
    arraySize = inf;
end

if (createFlag)
    h5create(fileName, datasetName, [inDims(1:end-1), arraySize], 'Datatype', 'single', 'ChunkSize', [inDims(1:end-1), chunkSize]);
end

h5write(fileName, datasetName, single(input), startLoc, inDims);

endInd = startLoc(end) + inDims(end) - 1;

