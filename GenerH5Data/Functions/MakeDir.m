function MakeDir(inputPath)

if (~exist(inputPath, 'dir'))
    mkdir(inputPath);
end