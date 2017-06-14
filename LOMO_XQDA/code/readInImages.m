function newImages = readInImages(images, imgWidth, imgHeight, imResizeMethod, options)
    n = size(images,4);
    READ_STD=1;
    READ_CENTRAL=2;
    READ_ALL=3;
    READ_DISTORT=4;
    newImages=zeros(imgHeight, imgWidth, 3, n, 'uint8');
    for i = 1 : n
        temp = squeeze(images(:,:,:,i));
        switch imResizeMethod
            case READ_STD
                newImages(:,:,:,i)=imageResizeStd(temp, imgWidth, imgHeight);
            case READ_CENTRAL
                newImages(:,:,:,i)=imageResizeCtrl(temp, imgWidth, imgHeight);
            case READ_ALL
                newImages(:,:,:,i)=imageResizeAll(temp, imgWidth, imgHeight);
            case READ_DISTORT
                t=imresize(temp, [imgHeight imgWidth]);
                newImages(:,:,:,i)=t;
        end
        if(options.retinexy)
            newImages(:,:,:,i) = Retinex(squeeze(newImages(:,:,:,i)));
        end

        %newImages(:,:,:,i) = imresize(temp,[imgHeight imgWidth]);   
    end
end