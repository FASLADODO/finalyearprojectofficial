%% CHANGE HOW TRAINING DONE NEED TO BE PAIRS, BUT SUBSET OF PAIRS.
%% TEST NEEDS TO BE MORE, OTHER PAIRS
%%IT IS NOT SENTENCES, THERE ARE NO IMAGES WITH >2 EXAMPLES FOR THE ID
function [personIds,descriptors] = ALEX(images,personIdsIn, options)
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
    else
        noImages=size(images,4);
    end
    if isfield(options,'imResizeMethod') && ~isempty(options.imResizeMethod) && isscalar(options.imResizeMethod) && isnumeric(options.imResizeMethod) && options.imResizeMethod > 0
        imResizeMethod = options.imResizeMethod;
        fprintf('Resizing method of images is %s.\n', resizeMethodNames{imResizeMethod});
    end
end
fprintf('Number of images extracting features from is %d.\n', noImages);
t0 = tic;
% Get GPU device information
%deviceInfo = gpuDevice;

% Check the GPU compute capability
%computeCapability = str2double(deviceInfo.ComputeCapability);
%assert(computeCapability >= 3.0, ...
 %   'This example requires a GPU device with compute capability 3.0 or higher.')

%% create image datastores
%{

%}

%{
switch imResizeMethod
    case READ_STD
        images=imageResizeStd(images);
    case READ_CENTRAL
        images=imageResizeCtrl(images);
    case READ_ALL
        images=imageResizeAll(images);
end
%}

%[imagesTrain,imagesTest]= splitEachLabel(imageStore,trainSplit);



%%Display 20 sample images
%idx= randperm(size(images,4));
%personIds=personIdsIn(idx(1:noImages));

[personIds,idx]=sort(personIdsIn);
personIds=personIds(1:noImages);
images=images(:,:,:,idx(1:noImages));
figure
title('Pre Zero centering and normalising input images')
for i = 1:min([20,noImages])
    subplot(4,5,i)
    
    %I = readimage(imagesTrain,i);
    imshow(squeeze(images(:,:,:,i)));
    drawnow
end

fprintf('Currently Zero centering and normalising input images \n')
for i= 1:noImages
    meany=im2double(repmat(mean(mean(squeeze(images(:,:,:,i)),1),2),size(images,1)));
    I=squeeze(images(:,:,:,i));
    images(:,:,:,i)=I-uint8(meany);
    images(:,:,:,i)=(squeeze(im2double(images(:,:,:,i)))./std(squeeze(im2double(images(:,:,:,i))),0,1))./std(squeeze(im2double(images(:,:,:,i))),0,2);
end
 



figure
title('Post Zero centering and normalising input images')
for i =  1:min([20,noImages])
    subplot(4,5,i)
    
    %I = readimage(imagesTrain,i);
    imshow(uint8(squeeze(images(:,:,:,i))));
    drawnow
end
fprintf('Images have been pre-processed. \n')
%% create net instance, get properties
split=int16(trainSplit*noImages);
%Images and ids have been sorted, need to generate separation to allow
%effective training of data
net = alexnet;

%% Create imagesTrain, imagesTest, personIdsTrain, personIdsTest from personIds, images
%imagesTrain=images(:,:,:,1:split);
%imagesTest=images(:,:,:,split+1:end);
            occur=0;
            indexes=[];
            idx=1;
            old=0;
            %%Construct sentencesTest and sentencesTrain based on settings,
            % Get indexes for train data
            switch options.sentenceSplit
                case 'pairs'
                        fprintf('Training data is all pairs that exist of sentenceData\n');
                        for i=1:length(personIds)
                            if(personIds(i)~=old && occur>1)
                                for p=1:2%occur
                                   indexes(idx)=i-p;
                                   idx=idx+1;                               
                                end      
                                occur=1;
                                old=personIds(i);
                            else
                                if(personIds(i)==old)
                                    occur=occur+1;
                                else
                                   old=personIds(i);
                                   occur=1;
                                end
                            end
                        end   
                case 'oneofeach'
                        fprintf('Training data is one of each of all sentence data');
                       for i=1:length(personIds)
                            if(personIds(i)~=old && occur>1)
                                
                                indexes(idx)=i-p;
                                idx=idx+1;
                                      
                                occur=1;
                                old=personIds(i);
                            else
                                if(personIds(i)==old)
                                    occur=occur+1;
                                else
                                   old=personIds(i);
                                   occur=1;
                                end
                            end
                        end
                    
                case 'oneofeach+'
                       fprintf('Training data is one of each of all sentence data + extras');
                       for i=1:length(personIds)
                            if(personIds(i)~=old && occur>1)
                                
                                indexes(idx)=i-p;
                                idx=idx+1;
                                      
                                occur=1;
                                old=personIds(i);
                            else
                                if(personIds(i)==old)
                                    occur=occur+1;
                                else
                                   old=personIds(i);
                                   occur=1;
                                end
                            end
                        end                   
                        indexes= setdiff([1:size(images,1)],indexes);
            end       
            %create sentencesTrain and sentencesTest
            %sentneceProcess is all current data in this configfile
            personIds2=categorical(personIds);
            imagesTrain=images(:,:,:,indexes);
            imagesIdsHot=hotCoding(personIds);
            imagesIdsTrain=personIds2(indexes);
            imagesIdsTrainHot=imagesIdsHot(indexes,:);
            testIndexes= setdiff([1:size(images,4)],indexes);
            imagesTest=images(:,:,:,testIndexes);
            'test ids'
            imagesIdsTest=personIds2(testIndexes)
            imagesIdsTestHot=imagesIdsHot(testIndexes,:);

            %% ONLY WORKS WITH PAIRS
            %%Reduce size of imagestrain so that all that exist with third
            %%in testData
            %%are included plus others up to 100
            %ismember(A,B) returns an array containing logical 1 (true) where the data in A is found in B.
            % PRESUMES IMAGETRAINSPLIT > OCCURENCE OF TRIPLES (WHICH IT
            % SHOULD BE)
            if(strcmp(options.sentenceSplit,'pairs'))
                trainIdsThird=find(ismember(imagesIdsTest, imagesIdsTrain));
                imagesSubIdsTrain=zeros(options.imageTrainSplit,1);
                imagesSubTrain=zeros(227,227,3,options.imageTrainSplit);
                imagesSubIdsTrain(1:length(trainIdsThird))=imagesIdsTrain(trainIdsThird);
                imagesSubTrain(:,:,:,1:length(trainIdsThird))=imagesTrain(:,:,:,trainIdsThird);
                imagesSubIdsTrain(length(trainIdsThird)+1:end)=imagesIdsTrain(1:options.imageTrainSplit-length(trainIdsThird));
                imagesSubTrain(:,:,:,length(trainIdsThird)+1:end)=imagesTrain(:,:,:,1:options.imageTrainSplit-length(trainIdsThird));
                imagesSubTest=imagesTest;
                imagesSubIdsTest=imagesIdsTest;
            %In this case do first 100, should have example of each, 
            else
                %imagesSubIdsTrain=zeros(options.imageTrainSplit,1);
                %imagesSubTrain=zeros(227,227,3,options.imageTrainSplit); 
                imagesSubIdsTrain=imagesIdsTrain(1:options.imagesTrainSplit);
                imagesSubTrain=imagesTrain(:,:,:,1:options.imagesTrainSplit);
                testSubIndexes=find(ismember(imagesSubIdsTrain,imagesIdsTest));
                imagesSubIdsTest=imagesIdsTest(testSubIndexes(1:options.imagesTrainSplit));
                imagesSubTest=imagesTest(:,:,:,testSubIndexes(1:options.imagesTrainSplit));
            end
%% Perform fine tuning
layersTransfer=net.Layers(1:end-3); %get last 3 layers to configure
%create new layer array by combining transferred with new layers
numClasses=length(unique(imagesSubIdsTrain));%size(imagesIdsTrainHot,1);%length(unique(imagesIdsTrain));personids
layers=[...
        layersTransfer
        fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
        softmaxLayer
        classificationLayer];
optionsNet=trainingOptions('sgdm',...
    'MiniBatchSize',5,...
    'MaxEpochs',5,...%10
    'InitialLearnRate',0.0001);
size(imagesSubTrain)
size(imagesSubIdsTrain)
%size(personIds)
size(imagesSubTest)
size(imagesSubIdsTest)
% labels are n-by-r numeric matrix, where n is the number of observations and r is the number of responses
netTransfer=trainNetwork(imagesSubTrain,categorical(imagesSubIdsTrain),layers,optionsNet);%layers char(imagesIdsTrain));
fprintf('Confusion matrix after fine tuning');
predictedLabels= classify(netTransfer, imagesSubTest);
%DONT BOTHER AS CATEGORICAL CHANGES VALUES AND TEST IMAGES ONLY HAVE 2
%EXAMPLES OF EACH ANYWAYS
'size imagesSubids'
size(imagesSubIdsTest)
'size predictedlabels'
size(predictedLabels)
%only compare relevant subsection of predicted labels
%resultIndexes=zeros(10,1);
%for i = 1:10%size(origLabels,2)
%	resultIndexes(i)= find(imagesIdsTestHot(:,i))
%end
%plotconfusion(categorical(imagesIdsTest),predictedLabels); NEED MATRIX OR
%CELL ARRAY
netTransfer.Layers
total=0;
predictedLabels
imagesSubIdsTest
for i = 1:length(predictedLabels)
    if(predictedLabels(i)==imagesSubIdsTest(i))
       total=total+1; 
    end
end
accuracy = total/numel(predictedLabels)
%plotconfusion(imagesIdsTestHot(resultIndexes,1:10),predictedLabels(resultIndexes,1:10));
% net = train(net,imagesTrain,imagesIdsTrain);%,'useParallel','yes','showResources','yes'
% NOT VALID ON ALEXNET
% Get prelim results of classifications in confusion matrix

%fprintf('Confusion matrix after fine tuning');
%testLabelPredictions = net(imagesTest);
%plotconfusion(imagesIdsTest,testLabelPredictions);
fprintf('net transfer layers');
netTransfer.Layers
%% extract Features: descriptors
layer = 'fc';
sz=sprintf('%d ', size(squeeze(images(:,:,:,1))));
fprintf('Input size is: 227 227 3 with zerocenter normalisation and layer input size is %s \n', sz)
fprintf('Now extracting training features');
trainingFeatures = activations(netTransfer,imagesTrain,layer);
sz=sprintf('%d ', size(trainingFeatures));
fprintf('Training features extracted, size: %s\n', sz)
fprintf('Now extracting test features');
testFeatures = activations(netTransfer,imagesTest,layer);
sz=sprintf('%d ', size(testFeatures));
fprintf('Test features extracted, size: %s\n', sz)

%% finishing, clear temp vars, create descriptors
descriptors=[trainingFeatures;testFeatures];

feaTime = toc(t0);
meanTime = feaTime / size(images, 4);
fprintf('ALEX feature extraction finished. Running time: %.3f seconds in total, %.3f seconds per image.\n', feaTime, meanTime);


end
function newImages=imageResizeStd(images)
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
end

function newImages=imageResizeCtrl(images)
        newImages=zeros(227,227,3,size(images,4));
        imgSize = [227, 227, 3];
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

function newImages=imageResizeAll(images)
         newImages=zeros(227,227,3,size(images,4));
         imgSize = [227, 227, 3];
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
                imageData=I(idx:imgSize(1)+idx,1:imgSize(2),1:imgSize(3));
            else  
                I=imresize(I,scaleX);
                idx=int16(((size(I,1)-imgSize(1))/2));
                imageData=I(idx:imgSize(1)+idx,1:imgSize(2),1:imgSize(3)); 
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



