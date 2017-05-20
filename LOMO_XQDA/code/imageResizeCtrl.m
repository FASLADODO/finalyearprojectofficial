function newImage=imageResizeCtrl(image, imageWidth, imageHeight)
        newImage=zeros(imageHeight,imageWidth,3);
        imgSize = [imageHeight, imageWidth, 3];
        
                I = image;
                scaleY=imgSize(1)/size(I,1);
                scaleX=imgSize(2)/size(I,2);
                if(scaleX>scaleY)
                    %image is smaller than input
                    if(scaleX>=1.0) %Y is now larger than it should be
                        I=imresize(I,scaleX);
                        idx=int16(abs((size(I,1)-imgSize(1))/2));
                        newImage(:,:,:)=I(idx:imgSize(1)+idx-1,1:imgSize(2),1:imgSize(3));
                    else  
                        I=imresize(I,scaleX);
                        idx=int16(abs((size(I,1)-imgSize(1))/2));
                        newImage(:,:,:)=I(idx:imgSize(1)+idx-1,1:imgSize(2),1:imgSize(3)); 
                    end
                else
                    I=imresize(I,scaleY);%scaled height so width, numcols wrong
                    %eg 41 20 so want 10-30
                    idx=int16(abs((size(I,2)-imgSize(2))/2));
                    newImage(:,:,:) = I(1:imgSize(1),idx:idx+imgSize(2)-1,1:imgSize(3)); 
                end    
                
end