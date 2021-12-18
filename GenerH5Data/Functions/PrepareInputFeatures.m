function [curInputs, curLabel] = PrepareInputFeatures(curImgsLDR, curExpo, curLabel, isTest)

if(~exist('isTest', 'var'))
    isTest = false;
end

numImgs = length(curImgsLDR);

curInLDR = curImgsLDR;

%%% handling the boundary regions where flow is not valid
nanInds1 = isnan(curInLDR{1});
curInLDR{1}(nanInds1) = LDRtoLDR(curInLDR{2}(nanInds1), curExpo(2), curExpo(1));

nanInds2 = isnan(curInLDR{3});
curInLDR{3}(nanInds2) = LDRtoLDR(curInLDR{2}(nanInds2), curExpo(2), curExpo(3));

if(~isTest)
    darkRef = curInLDR{2} < 0.5;
    badRef = (darkRef & nanInds2) | (~darkRef & nanInds1);
    curLabel(badRef) = (LDRtoHDR(curInLDR{2}(badRef), curExpo(2)));
end


%%% concatenating inputs in the LDR and HDR domains
curLabel = (curLabel);
curInputs = (curInLDR{1});
for k = 2 : numImgs
    curInputs = cat(3, curInputs, (curInLDR{k}));
end

for k = 1 : numImgs
    curInputs = cat(3, curInputs, (LDRtoHDR(curInLDR{k}, curExpo(k))));
end