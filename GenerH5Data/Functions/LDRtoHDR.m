function out = LDRtoHDR(input, expo)

global gamma;

input = Clamp(input, 0, 1);
% input = single(input);
out = (input).^gamma;
out = out ./ expo;

