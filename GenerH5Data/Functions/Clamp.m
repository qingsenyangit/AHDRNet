function out = Clamp(input, a, b)

out = input;

out(out < a) = a;
out(out > b) = b;