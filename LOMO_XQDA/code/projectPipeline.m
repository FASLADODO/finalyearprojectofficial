%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Final Year Project Pipeline to handle all available classification,
%%feature extraction methods automatically


%%Feature extraction works with auto detect file pre-existence and forcing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Notes
%Matlab uses 1-indexing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear; clc;
close all;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Setup- set directories, number of experiment repeats
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

imgDir = '../images/';
evalDir = '../evaluators/';
classifyDir = '../classifiers/';
featDir = '../featureExtractors/';
featuresDir='../data/';
featureExts='*.mat';
resultsDir='../results/';
addpath(imgDir);
addpath(featDir);
addpath(classifyDir);
addpath(evalDir);

%% Experiment parameters
numFolds=10; %Number of times to repeat experiment
numRanks = 100;
READ_STD=1;
READ_CENTRAL=2;
READ_ALL=3;
options.imResizeMethod=READ_ALL;
options.trainSplit=0.6;

options.noImages=0;%if 0 then all run


%%Features
LOMO_F=1;
ALEX_F=2;
VGG_F=3;
%%Classifiers
XQDA_F=1;
%%Which feature extractors to run
%%Which classifiers to run
featureExtractors= [{LOMO_F, @LOMO};{ALEX_F, @ALEX};{VGG_F, @VGG}];%%,{MACH, @MACH}
featureImgDimensions=[128,48; 227,227; 224,224]; %100 40
featureName={'LOMO.mat', 'ALEX.mat', 'VGG.mat'};
imgType={'Std','Ctrl','All'};

featureForce=true; 
featureExtractorsRun=[VGG_F];%LOMO_F

classifiers= [{XQDA_F, @XQDA}];
classifiersRun=[XQDA_F];
classifierName={'XQDA'};

features=[];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Import all images
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
imgList = dir([imgDir, '*.png']);%[imgDir, '*.png']
n = length(imgList);

%% Allocate memory
info = imfinfo([imgDir, imgList(1).name]);
imgWidth=40;
imgHeight=100;


%% read categories
person_ids=zeros(n,1);
for i=1:n
   name=imgList(i).name;%Format image=06_set=3_id=0001
   temp= strsplit(name,{'image=','_set=','_id=','.png'});
   person_ids(i)= str2double(strcat(temp(3),temp(4)));%str2double()
   %person_ids_char(i)= char(strcat(temp(3),temp(4)));
end
%options.noImages=n;

images = zeros(imgHeight,imgWidth, 3, n, 'uint8');
%% read images
for i = 1 : n
    temp = imread([imgDir, imgList(i).name]);
    images(:,:,:,i) = imresize(temp,[imgHeight imgWidth]);   
end

%% image datastore for pre-trained deep learning models
imds = imageDatastore(imgDir);
%{
for i=1:n
   name=imgList(i).name;%Format image=06_set=3_id=0001
   temp= strsplit(name,{'image=','_set=','_id=','.png'});
   %Labels is an empty cell array
   %str cat gives string array
   imds.Labels(i)= strcat(temp(3),temp(4));
end
%}
%imds.Labels=cellstr(person_ids_char);
imds.Labels=cellstr(char(person_ids));
[imagesTrain,imagesTest]= splitEachLabel(imds,0.6);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Extract features using all feature extraction methods
%%Store in data directory
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%if feature set doesn't already exist

%%Perform feature extraction
%%Save to data folder
disp('Checking if features to Extract already exist or forced')
for i=1:length(featureExtractorsRun)
    %Check if features already exist 
    featureList=dir([featuresDir, '*.mat']);
    featuresAvail=[featureList.name];
    currFeatureName=cell2mat(featureName(featureExtractorsRun(i)));
    config=sprintf('%d_%d_%d_',options.imResizeMethod,int16(options.trainSplit*100),options.noImages);
    currFeatureName=strrep(strcat(config,currFeatureName),' ', '');
    %Run feature extraction function
    %If being forced or features Available doesnt already exist
    if(featureForce || isequal(strfind(featuresAvail,currFeatureName),[]))
        idx=find(cell2mat(featureExtractors(:,1))==featureExtractorsRun(i),1);
        %Could do error checking here to test match exists: 1x0
        %featureID= cell2mat(featureExtractors(u,1));
        featureFunct= cell2mat(featureExtractors(idx,2));
        fprintf('Extracting current feature %s, place in data directory \n',currFeatureName)
        features=[];
        personIds=[];
        %Load images according to size of feature extraction process
        imgWidth=featureImgDimensions(featureExtractorsRun(i),2);
        imgHeight=featureImgDimensions(featureExtractorsRun(i),1);
        %images = zeros(imgHeight,imgWidth, 3, n, 'uint8');
        images = readInImages(imgDir, imgList, imgWidth, imgHeight, options.imResizeMethod);
        %% read images
        %for i = 1 : n
        %   temp = imread([imgDir, imgList(i).name]);
         %  images(:,:,:,i) = imresize(temp,[imgHeight imgWidth]);   
        %end
       

        %RandonPerm depending on noImages, need to keep associated
        %order of personIds or worthless
        [personIds, features]=featureFunct(images,person_ids, options); %(:,:,:,1:options.noImages) done inside function 
            

        
        save(char(strcat(featuresDir,currFeatureName)),'features', 'personIds');
        %{
        for u=1:size(featureExtractors,1)
            featureID= cell2mat(featureExtractors(u,1));
            featureFunct= cell2mat(featureExtractors(u,2));       
            if(featureExtractorsRun(i)==featureID)
                fprintf('Extracting current feature %s',currFeatureName)
                %features(i)=featureFunct(images);
            end
        end
        %}
    else
      fprintf('Already exists. Not extracting current feature %s, config %d %d %d \n',currFeatureName,options.imResizeMethod,options.trainSplit,options.noImages)
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Create natural language-->matrix projections
%%Store in descriptions matrix with accompanying descriptions labels
%%matrices
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Perform classification on all extracted features
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%Get features
%%gets data on all files in directory, store in array of structs
%% probFea and galFea
numImages=0;
featureList = dir(strcat(featuresDir,featureExts));%data/*.mat
%[featuresDir, '*.mat']
featuresAvail=[featureList.name];
%%Loads extracted features into arrays
%for i= 1: size(featuresList,1)
for i=1:length(featureExtractorsRun)
    currFeatureName=cell2mat(featureName(featureExtractorsRun(i)));
    config=sprintf('%d_%d_%d_',options.imResizeMethod,int16(options.trainSplit*100),options.noImages);
    currFeatureName=strrep(strcat(config,currFeatureName),' ', '');
    %%loads variables called descriptors from file
    %%featureList(1,i) as 1x1 struct
    %fileName=strcat(featuresDir,featuresList(i).name);
    
    if(~isequal(strfind(featuresAvail,currFeatureName),[]))
        fprintf('Currently loading features %s into matrices \n',currFeatureName);
        load(char(strcat(featuresDir,currFeatureName)));%originally saved as features, personIds
        if(size(features,1) ~= length(imgList))
            descriptors=features.';
        else
            descriptors=features;
        end
        
        
        % Sort person Ids and images so can be split effectively for
        % matching
        [personIds,idx] = sort(personIds);
        descriptors=descriptors(idx,:);
        %now both sorted ascending
        
        
        numImages= int16(size(descriptors,1)/2);
        personIds=[personIds(1:2:end);personIds(2:2:end)];
        descriptors=[descriptors(1:2:end,:); descriptors(2:2:end,:)];
        
        galFea(i,:,:) = descriptors(1 : numImages, :);
        probFea(i,:,:) = descriptors(numImages + 1 : end, :);
        classLabelGal(i,:)=personIds(1:numImages);
        classLabelProb(i,:)=personIds(numImages+1:end);
        clear descriptors
        clear personIds
    else
        fprintf('Could not load features %s into matrices as folder didnt exist \n',currFeatureName);
    end
end
%%For all extracted features 
%%For all classification techniques
%%Get results
%%Features split between galFea, probFea, 
%%Feature labels split between classLabelGal, classLabelFea

noTests=length(classifiersRun)*size(galFea,1);%%galfea i is number rows?
cms = zeros(noTests, numFolds, numRanks);

%%Select classifiers want to run
for i=1:length(classifiersRun)
        idx=find(cell2mat(classifiers(:,1))==classifiersRun(i),1);
        currClassifierId=cell2mat(classifiers(idx,1));
        currClassifierFunct=cell2mat(classifiers(idx,2));
        currClassifierName=cell2mat(classifierName(currClassifierId));
        fprintf('Currently running classifier %s \n',currClassifierName)
            %%For every set of features
            for ft=1:size(galFea,1)
                %Repeat classification process numFolds times
                for iter=1:numFolds
                    
                    p = randperm(numImages);
                    galFea1 = squeeze(galFea( ft,p(1:int16(numImages/2)), : ));
                    probFea1 = squeeze(probFea(ft, p(1:int16(numImages/2)), : ));
                    classLabelGal1=squeeze(classLabelGal(ft,p(1:int16(numImages/2))));
                    classLabelProb1=squeeze(classLabelProb(ft,p(1:int16(numImages/2))));
                    
                    
                    t0 = tic;
                    %[W, M] = XQDA(galFea1, probFea1, (1:int16(numImages/2))', (1:int16(numImages/2))');
                    [W, M] = XQDA(galFea1, probFea1, classLabelGal1', classLabelProb1');

                    %{
                    %% if you need to set different parameters other than the defaults, set them accordingly
                    options.lambda = 0.001;
                    options.qdaDims = -1;
                    options.verbose = true;
                    [W, M] = XQDA(galFea1, probFea1, (1:numImages/2)', (1:numImages/2)', options);
                    %}

                    clear galFea1 probFea1
                    trainTime = toc(t0);
                    %Squeeze removes singleton dimensions
                    galFea2 = squeeze(galFea(ft, p(int16(numImages/2)+1 : end), : ));
                    probFea2 = squeeze(probFea(ft, p(int16(numImages/2)+1 : end), : ));
                    classLabelGal2=squeeze(classLabelGal(ft,p(int16(numImages/2)+1 : end)));
                    classLabelProb2=squeeze(classLabelProb(ft,p(int16(numImages/2)+1 : end)));
                    
                    t0 = tic;
                    %Produces score matrix of all distances between probe
                    %and gallery images
                    dist = MahDist(M, galFea2 * W, probFea2 * W);
                    clear galFea2 probFea2 M W
                    matchTime = toc(t0);      

                    fprintf('Fold %d: ', iter);
                    fprintf('Training time: %.3g seconds. ', trainTime);    
                    fprintf('Matching time: %.3g seconds.\n', matchTime); 
                    %CMS is for every feature set, repeated 10 times, 100 ranks.
                    %Different cms for every classifier
                    %Input is matrix showing distance vs all gallery &
                    %probe images
                    %Next two are labels
                    %cms(ft, iter,:) = EvalCMC( -dist, 1 : int16(numImages / 2), 1 : int16(numImages / 2), numRanks );
                    cms(ft, iter,:) = EvalCMC( -dist, classLabelGal2, classLabelProb2, numRanks );
                    clear dist           

                    fprintf(' Rank1,  Rank5, Rank10, Rank15, Rank20\n');
                    fprintf('%5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%\n\n', cms(ft, iter,[1,5,10,15,20]) * 100);

                end
                %Mean for every feature set, classifier combination
                meanCms = mean(squeeze(cms(ft,:,:)));
                figure
                plot(1 : numRanks, meanCms)
                currFeatureName=cell2mat(featureName(featureExtractorsRun(ft)));
                config=sprintf('%d_%d_%d',options.imResizeMethod,int16(options.trainSplit*100),options.noImages);
                
                title(sprintf('CMS Curve for Classifier %s, feature set %s and settings %s', currClassifierName, currFeatureName, config))
                xlabel('No. Ranks of ordered Gallery Images') % x-axis label
                ylabel('% Gallery Images that contain match within that rank') % y-axis label

                csvFileName=strcat(resultsDir,currClassifierName,'_',currFeatureName,'_', config);
                csvwrite(csvFileName,meanCms)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
                %%type csvlist.dat

                fprintf('The average performance:\n');
                fprintf(' Rank1,  Rank5, Rank10, Rank15, Rank20\n');
                fprintf('%5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%\n\n', meanCms([1,5,10,15,20]) * 100);
            end
       % end
   % end
end









