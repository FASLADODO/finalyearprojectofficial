function newImages = readInImages(imgDir, imgList, imgWidth, imgHeight, imResizeMethod)
    n = length(imgList);
    READ_STD=1;
    READ_CENTRAL=2;
    READ_ALL=3;
    newImages=zeros(imgHeight, imgWidth, 3, n, 'uint8');
    for i = 1 : n
        temp = imread([imgDir, imgList(i).name]);
        switch imResizeMethod
            case READ_STD
                newImages(:,:,:,i)=imageResizeStd(temp, imgWidth, imgHeight);
            case READ_CENTRAL
                newImages(:,:,:,i)=imageResizeCtrl(temp, imgWidth, imgHeight);
            case READ_ALL
                newImages(:,:,:,i)=imageResizeAll(temp, imgWidth, imgHeight);
        end

        %newImages(:,:,:,i) = imresize(temp,[imgHeight imgWidth]);   
    end
end