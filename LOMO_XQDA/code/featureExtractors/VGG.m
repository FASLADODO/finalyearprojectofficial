
function descriptors = VGG(images, options)
%% function Descriptors = MACH(images, options)
% Function for the machine learning feature extraction
%
% Input:
%   <images>: a set of n RGB color images. Size: [h, w, 3, n]

% Output:
%   descriptors: the extracted LOMO descriptors. Size: [d, n]
% 
% Example:
%     I = imread('../images/000_45_a.bmp');
%     descriptor = LOMO(I);
READ_STD=1;
READ_CENTRAL=2;
READ_ALL=3;
resizeMethodNames={'Standard','Central', 'All'};

%% set parameters, check system
if nargin >= 2
    if isfield(options,'trainSplit') && ~isempty(options.trainSplit) && isscalar(options.trainSplit) && isnumeric(options.trainSplit) && options.trainSplit > 0
        trainSplit = options.trainSplit;
        fprintf('Training percentage of images is %d.\n', trainSplit);
    end
    if isfield(options,'noImages') && ~isempty(options.noImages) && isscalar(options.noImages) && isnumeric(options.noImages) 
        if(options.noImages==0)
            noImages=size(images,4);
        else
            noImages = options.noImages;
        end      
        fprintf('Number of images used is %d.\n', noImages);
    end
    if isfield(options,'imResizeMethod') && ~isempty(options.imResizeMethod) && isscalar(options.imResizeMethod) && isnumeric(options.imResizeMethod) && options.imResizeMethod > 0
        imResizeMethod = options.imResizeMethod;
        fprintf('Resizing method of images is %s.\n', resizeMethodNames{imResizeMethod});
    end
end
fprintf('Number of images extracting features from is %d.\n', size(images,4));
t0 = tic;
% Get GPU device information
%deviceInfo = gpuDevice;

% Check the GPU compute capability
%computeCapability = str2double(deviceInfo.ComputeCapability);
%assert(computeCapability >= 3.0, ...
 %   'This example requires a GPU device with compute capability 3.0 or higher.')

%% create image datastores
%%Display 20 sample images
figure
for i = 1:min([20,noImages])
    subplot(4,5,i)

    %I = readimage(imagesTrain,i);
    imshow(squeeze(images(:,:,:,i)));
    drawnow
end

%noImages=size(images,4);
images=images(:,:,:,1:noImages);
%{
switch imResizeMethod
    case READ_STD
        temp_images=imageResizeStd(images);
    case READ_CENTRAL
        temp_images=imageResizeCtrl(images);
    case READ_ALL
        temp_images=imageResizeAll(images);
end

images=temp_images;
%}
split=int16(trainSplit*noImages);
imagesTrain=images(:,:,:,1:split);
imagesTest=images(:,:,:,split+1:end);
%[imagesTrain,imagesTest]= splitEachLabel(imageStore,trainSplit);



%%Display 20 sample images
figure
for i = 1:min([20,noImages])
    subplot(4,5,i)

    %I = readimage(imagesTrain,i);
    imshow(squeeze(images(:,:,:,i)));
    drawnow
end

%% create net instance, get properties
net = vgg16;

%% extract Features: descriptors
layer = 'fc8';
trainingFeatures = activations(net,imagesTrain,layer);
sz=sprintf('%d ', size(trainingFeatures));
fprintf('Training features extracted, size: %s\n', sz)
testFeatures = activations(net,imagesTest,layer);
sz=sprintf('%d ', size(testFeatures));
fprintf('Test features extracted, size: %s\n', sz)
%% finishing, clear temp vars, create descriptors
descriptors=[trainingFeatures;testFeatures];
%rows dictate feature lists
%descriptors(1 : numImages, :);
sz=sprintf('%d ', size(descriptors));
fprintf('All features extracted, to be saved in .mat format size: %s\n', sz)

feaTime = toc(t0);
meanTime = feaTime / size(images, 4);
fprintf('VGG feature extraction finished. Running time: %.3f seconds in total, %.3f seconds per image. \n', feaTime, meanTime);


end

function newImages=imageResizeAll(images)
        newImages=zeros(224,224,3,size(images,4));
        imgSize = [224, 224, 3];
        for i=1:size(images,4)
                I = squeeze(images(:,:,:,i));
                scaleY=imgSize(1)/size(I,1);
                scaleX=imgSize(2)/size(I,2);
                if(scaleX>scaleY)
                    %image is smaller than input
                   % if(scaleX>=1.0) %Y is now larger than it should be
                        I=imresize(I,scaleY);
                        size(I)
                        idx=int16(((imgSize(2)-size(I,2))/2));
                        size(newImages(:,idx:idx+size(I,2)-1,:,i))
                        size(I(1:imgSize(1),:,:))
                        newImages(:,idx:idx+size(I,2)-1,:,i)=I(1:imgSize(1),:,:);
                    %else  
                       % I=imresize(I,scaleY);
                       % idx=int16(((imgSize(2)-size(I,2))/2));
                        %newImages(:,:,:,i)=I(idx:imgSize(1)+idx,1:imgSize(2),1:imgSize(3)); 
                    %end
                else%scaleY>scaleX
                    I=imresize(I,scaleX);%scaled width so height, numcols wrong
                    %eg 41 20 so want 10-30
                    idx=int16(((imgSize(1)-size(I,1))/2));
                    newImages(idx:idx+size(I,1)-1,:,:,i) = I(:,1:imgSize(2),:);
                end    
        end        
end


function newImages=imageResizeStd(images)
% 224x224x3 images with 'zerocenter' normalization
    


    %net = vgg16;
    %imgSize=[net.normalization.imageSize(1:2),3];
    imgSize=[224,224,3];
    newImages=zeros(224,224,3,size(images,4));
    for i=1:size(images,4)
            I = squeeze(images(:,:,:,i));
            scaleY=imgSize(1)/size(I,1);
            scaleX=imgSize(2)/size(I,2);
            if(scaleX>scaleY)
                I=imresize(I,scaleX);
            else
                I=imresize(I,scaleY);
            end
            %Resize images for net
            I = I(1:imgSize(1),1:imgSize(2),1:imgSize(3));
            meany=uint8(repmat(mean(mean(I)),224));
            normal=(I-meany);
            newImages(:,:,:,i) =normal;%((normal/std(I));
        
        %{
        newImages(:,:,:,i) = single(images(:,:,:,i)); % note: 255 range
        newImages(:,:,:,i) = imresize(newImages(:,:,:,i), net.normalization.imageSize(1:2)) ;
        newImages(:,:,:,i) = newImages(:,:,:,i) - net.normalization.averageImage ;
        %}
    end
    %{
        newImages=zeros(227,227,3,size(images,4));
        imgSize = [227, 227, 3];
        for i=1:size(images,4)
            I = squeeze(images(:,:,:,i));
            scaleY=imgSize(1)/size(I,1);
            scaleX=imgSize(2)/size(I,2);
            if(scaleX>scaleY)
                I=imresize(I,scaleX);
            else
                I=imresize(I,scaleY);
            end
            %Resize images for net
            newImages(:,:,:,i) = I(1:imgSize(1),1:imgSize(2),1:imgSize(3));
        end 
       %}
end
    %Leftmost
    function imageData= readAlexNetStd(imageName)
        imgSize = [227, 227, 3];
        I = imread(imageName);
        scaleY=imgSize(1)/size(I,1);
        scaleX=imgSize(2)/size(I,2);
        if(scaleX>scaleY)
            I=imresize(I,scaleX);
        else
            I=imresize(I,scaleY);
        end

        %Resize images for net
        imageData = I(1:imgSize(1),1:imgSize(2),1:imgSize(3)); 
    end

function newImages=imageResizeCtrl(images)
        newImages=zeros(224,224,3,size(images,4));
        imgSize = [224, 224, 3];
        for i=1:size(images,4)
                I = squeeze(images(:,:,:,i));
                scaleY=imgSize(1)/size(I,1);
                scaleX=imgSize(2)/size(I,2);
                if(scaleX>scaleY)
                    %image is smaller than input
                    if(scaleX>=1.0) %Y is now larger than it should be
                        I=imresize(I,scaleX);
                        idx=int16(((size(I,1)-imgSize(1))/2));
                        newImages(:,:,:,i)=I(idx:imgSize(1)+idx-1,1:imgSize(2),1:imgSize(3));
                    else  
                        I=imresize(I,scaleX);
                        idx=int16(((size(I,1)-imgSize(1))/2));
                        newImages(:,:,:,i)=I(idx:imgSize(1)+idx-1,1:imgSize(2),1:imgSize(3)); 
                    end
                else
                    I=imresize(I,scaleY);%scaled height so width, numcols wrong
                    %eg 41 20 so want 10-30
                    idx=int16(((size(I,2)-imgSize(2))/2));
                    newImages(:,:,:,i) = I(1:imgSize(1),idx:idx+imgSize(2)-1,1:imgSize(3)); 
                end    
        end        
end
    %Leftmost
    function imageData= readAlexNetCentral(imageName)

        imgSize = [227, 227, 3];
        I = imread(imageName);
        scaleY=imgSize(1)/size(I,1);
        scaleX=imgSize(2)/size(I,2);
        %image scalex largest
        if(scaleX>scaleY)
            %image is smaller than input
            if(scaleX>=1.0) %Y is now larger than it should be
                I=imresize(I,scaleX);
                idx=int16(((size(I,1)-imgSize(1))/2));
                imageData=I(idx:imgSize(1)+idx-1,1:imgSize(2),1:imgSize(3));
            else  
                I=imresize(I,scaleX);
                idx=int16(((size(I,1)-imgSize(1))/2));
                imageData=I(idx:imgSize(1)+idx-1,1:imgSize(2),1:imgSize(3)); 
            end
        else
            I=imresize(I,scaleY);%scaled height so width, numcols wrong
            %eg 41 20 so want 10-30
            idx=int16(((size(I,2)-imgSize(2))/2));
            imageData = I(1:imgSize(1),idx:idx+imgSize(2),1:imgSize(3)); 
        end    
    end
%{
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
iMAGE cATEGORY CLASSIFICATION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Get GPU device information
%deviceInfo = gpuDevice;

% Check the GPU compute capability
%computeCapability = str2double(deviceInfo.ComputeCapability);
%assert(computeCapability >= 3.0, ...
 %   'This example requires a GPU device with compute capability 3.0 or higher.')

url = 'http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz';
% Store the output in a temporary folder
outputFolder = fullfile(tempdir, 'caltech101'); % define output folder
if ~exist(outputFolder, 'dir') % download only once
    disp('Downloading 126MB Caltech101 data set...');
    untar(url, outputFolder);
end
rootFolder = fullfile(outputFolder, '101_ObjectCategories');
categories = {'airplanes', 'ferry', 'laptop'};
imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');
%imageDatastore labels is empty if not specified
tbl = countEachLabel(imds)
%%%%%%%%%%%%Balance out no. image labels for training
%Equal positive and negative
minSetCount = min(tbl{:,2}); % determine the smallest amount of images in a category

% Use splitEachLabel method to trim the set.
imds = splitEachLabel(imds, minSetCount, 'randomize');
% Find the first instance of an image for each category
airplanes = find(imds.Labels == 'airplanes', 1);

% Notice that each set now has exactly the same number of images.
countEachLabel(imds)

%}



