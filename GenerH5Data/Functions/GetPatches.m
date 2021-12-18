function [patches,count] = GetPatches(input, patchSize, stride, t)

[height, width, depth] = size(input);

numPatches = (floor((width-patchSize)/stride)+1)*(floor((height-patchSize)/stride)+1);
patches = zeros(patchSize, patchSize, depth, numPatches, 'single');

count = 0;
thres = 1/2 * patchSize * patchSize;
for iX = 1 : stride : width - patchSize + 1
    for iY = 1 : stride : height - patchSize + 1
        if sum(sum(t(iY:iY+patchSize-1, iX:iX+patchSize-1, 1))) < thres
           continue; 
        end
        count = count + 1;
        patches(:, :, :, count) = input(iY:iY+patchSize-1, iX:iX+patchSize-1, :);
    end
end