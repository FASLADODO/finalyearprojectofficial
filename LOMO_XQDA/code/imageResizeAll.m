function newImage=imageResizeAll(image, imageWidth, imageHeight)
         newImage=zeros(imageHeight,imageWidth,3);
         imgSize = [imageHeight, imageWidth, 3];
        
                I = image;
                scaleY=imgSize(1)/size(I,1);
                scaleX=imgSize(2)/size(I,2);
                
                    if(scaleX>scaleY)
                        %image is smaller than input
                       % if(scaleX>=1.0) %Y is now larger than it should be
                            I=imresize(I,scaleY);
                            
                            idx=int16(abs((imgSize(2)-size(I,2))/2))+1;

                            %size(newImages(:,idx:idx+size(I,2)-1,:,i))
                            %size(I(1:imgSize(1),:,:))
                            newImage(:,idx:idx+size(I,2)-1,:)=I(1:imgSize(1),:,:);
                        %else  
                           % I=imresize(I,scaleY);
                           % idx=int16(((imgSize(2)-size(I,2))/2));
                            %newImages(:,:,:,i)=I(idx:imgSize(1)+idx,1:imgSize(2),1:imgSize(3)); 
                        %end
                    else%scaleY>scaleX
                        I=imresize(I,scaleX);%scaled width so height, numcols wrong
                        %eg 41 20 so want 10-30
                        idx=int16(abs((imgSize(1)-size(I,1))/2))+1;
                        newImage(idx:idx+size(I,1)-1,:,:) = I(:,1:imgSize(2),:);
                    end    
 
                  
end