function frame2gif(volume,videoname,delaytime,resizeRate,height)
if exist('delaytime','var')~=1
    delaytime=0.5;
end
if exist('resizeRate','var')~=1
    resizeRate=1;
end
if exist('height','var')~=1
    height=0;
end

nframes=size(volume,4);

for i=1:nframes
    im=volume(:,:,:,i);
    if height~=0
        [h,w,nchannels]=size(im);
        resizeRate=height/h;
    end
    if resizeRate<1
        im=imresize(imfilter(im,fspecial('gaussian',5,0.7),'same','replicate'),resizeRate,'bicubic');
    else if resizeRate>1
            im=imresize(im,resizeRate,'bicubic');
        end
    end
    [X,map]=rgb2ind(im,256);
    if i==1
        imwrite(X,map,videoname,'DelayTime',delaytime,'LoopCount',Inf);
    else
        imwrite(X,map,videoname,'WriteMode','append','DelayTime',delaytime);
    end
end