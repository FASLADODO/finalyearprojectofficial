function newImage=imageResizeStd(image, imageWidth, imageHeight)
        newImage=zeros(imageHeight,imageWidth,3);
        imgSize = [imageHeight,imageWidth,3];
        
            I = image;
            scaleY=imgSize(1)/size(I,1);
            scaleX=imgSize(2)/size(I,2);
            if(scaleX>scaleY)
                I=imresize(I,scaleX);
            else
                I=imresize(I,scaleY);
            end
            %Resize images for net
            newImage(:,:,:) = I(1:imgSize(1),1:imgSize(2),1:imgSize(3));
                
end