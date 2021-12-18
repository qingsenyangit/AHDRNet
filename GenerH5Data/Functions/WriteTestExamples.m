function WriteTestExamples(inputs, label, savePath)

chunksz = 10;

startloc = [1, 1, 1];

% SaveHDF(savePath, '/IN', single(inputs), PadWithOne(size(inputs), 3), startloc, chunksz, true);
% SaveHDF(savePath, '/GT', single(label) , PadWithOne(size(label), 3) , startloc, chunksz, true);
hdf5write(savePath, '/GT', single(label),  '/IN', single(inputs));