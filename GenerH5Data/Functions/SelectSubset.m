function inds = SelectSubset(input)

global param;
patchSize = param.patchSizeforthresh;

% maxTH = 0.8;
% minTh = 0.2;
% 
% thresh = 0.5 * patchSize * patchSize * 3;
% 
% badInds = input > maxTH | input < minTh;
% 
% inds = sum(sum(sum(badInds, 1), 2), 3) > thresh;
% inds = find(inds == 1);



I1 = double(rgb2gray(input(:,:,1:3))); 
I2 = double(rgb2gray(input(:,:,4:6)));
I3 = double(rgb2gray(input(:,:,7:9)));

t1 = abs(I3-I2);
t1(t1>0.04) =1;
t1(t1<=0.04) =0;

t2 = abs(I1-I2);
t2(t2>0.04) =1;
t2(t2<=0.04) =0;

t = t1 + t2;
t(t>=1) = 1;

thresh = 1/9 * patchSize * patchSize;

inds = sum(sum(t, 1), 2) > thresh;
inds = find(inds == 1);