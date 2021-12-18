function adjusted = AdjustExposure(imgs, expoTimes)

%%% This function adjust the exposure of all the images to the image with
%%% highest exposure.

numImgs = length(imgs);
numExposures = length(expoTimes);

assert(numImgs == numExposures, 'The number of input images is not equal to the number of exposures');

adjusted = cell(1, numImgs);
maxExpo = max(expoTimes);

for imgInd = 1 : numImgs
    adjusted{imgInd} = LDRtoLDR(imgs{imgInd}, expoTimes(imgInd), maxExpo);
end