function out = HDRtoLDR(input, expo)

global gamma;

input = single(input) * expo;
input = Clamp(input, 0, 1);
out = input.^(1/gamma);


