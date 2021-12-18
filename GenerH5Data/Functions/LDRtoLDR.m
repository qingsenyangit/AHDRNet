function B = LDRtoLDR(A, expA, expB)

Radiance = LDRtoHDR(A, expA);
B = HDRtoLDR(Radiance, expB);




