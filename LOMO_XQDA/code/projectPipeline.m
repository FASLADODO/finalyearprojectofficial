%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Final Year Project Pipeline to handle all available classification,
%%feature extraction methods automatically
%NEED PCA FOR WORDS, VGG NET FINISH, ANOTHER SENTENCE-IMAGE METHOD, SCRIPT TO SHOW CMS GRPAHS 

%have added methods to skip alex, imageSentence twosentences and autoencode
%sentneces net generation
%image features only extracted if dont already exist, (TO CHANGE NET
%STRUCTURE GET NEW FILES)
%sentences only extract if dont exist, MIGHT NEED SENTENCEFORCE OVERRIDE
%LATER
%mean and mode run by auto as quick
%classification only performed if results dont already exist, 
%determined no case where wont need to run net but will classification
%process?

%% Where to save 
%data/images, data/sentences
%results/images, results/sentences, results/imageSentences
%nets/images, nets/sentences, , nets/classification

%% Results format  
% classifier, settings, image, settings, sentences settings
% classifier, settings, image settings
% classifier, settings, sentences, settings

%% Image feature settings %create other layer configurations using alex2, vgg2 etc yup
%lomo, resizemethod, imagetrainsplit , imageOptions.noImagesyup 
%alex, resizemethod, imagesTrainSplit, imageOptions.noImages yup
%vgg, resizemethod, imagesTrainSplit, imageOptions.noImages yup

%% Sentence feature settings - created via bash script already noted like this yup
% window, threshold, size (wordvec processing), 
%normalise, mode (python processing), 
% net format -->which autoencodesentences1/2, autoencodelevel,
% sentencesplit, hiddensize1, yup
%hiddensize2, maxepoch1 maxepoch2, maxepoch3, numSentences yup

%% classifier settings continue with twoChannels2 etc to change structure
%xqda,   pca/first
%twoChannels-pca/first, epochs, learning rate, falsepositiveratio, 

%% Net settings
%alex imagesTrainSplit imagesresizemethod 
%vgg imagesTrainSplit imagesresizemethod
%twoChannel  pca/first, epochs, learning rate, falsepositiveratio,
%autoEncodeSentences autoencodelevel, sentencesplit, hiddensize1,
%hiddensize2, maxepoch1 maxepoch2, maxepoch3, numSentences


%% Change sentences so file search based on settings, not filenames

%%Feature extraction works with auto detect file pre-existence and forcing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Notes
%Matlab uses 1-indexing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear; clc;
close all;
set(0,'DefaultTextInterpreter','none');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Setup- set directories, number of experiment repeats
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

imgDir = '../images/';
evalDir = '../evaluators/';
classifyDir = '../classifiers/';
sentencesDir='../../word2vec/trunk/matlab_sentence_vectors/';
imageFeaturesDir='../featureExtractors/images/';
sentenceFeaturesDir='../featureExtractors/sentences/';
featDir = '../featureExtractors/';
featuresDir='../data/';

featureExts='*.mat';
resultsDir='../results/';

addpath(imgDir);
addpath(featDir);
addpath(classifyDir);
addpath(evalDir);

%% Experiment parameters
numFolds=2; %Number of times to repeat experiment
numRanks = 100; %Number of ranks to show for
READ_STD=1;
READ_CENTRAL=2;
READ_ALL=3;
READ_DISTORT=4;
imageOptions.noImages=0;
imageOptions.imResizeMethod=READ_DISTORT;
imageOptions.imageTrainSplit=1000;
imageOptions.imageSplit='pairs'; %'oneofeach' 'oneofeach+' 
imageOptions.trainLevel=3; %autoEncode3 autoencoder level
imageOptions.hiddensize1=500;%199 1000
imageOptions.hiddensize2=100;%100 500
imageOptions.maxepoch1=100;
imageOptions.maxepoch2=50;
imageOptions.maxepoch3=100;

imageOptions.retinexy=true;
imageOptions.width=40;
imageOptions.height=40;



%options.noImages=0;%if 0 then all run
%options.featureExtractionMethod='AUTOENCODE3';%AUTOENCODE2, LOMO
options.falsePositiveRatio=1;
options.dimensionMatchMethod='pca'; %first pca FIRST USED WHEN COMPOSING NEURAL NETWORKS
options.testSize=200; %used for twoChannel, as matches go to 16,000,000 otherwise
options.hiddensize1=40;%199 1000 %sentences are size 40, so total is 80 if force match (but dont have to necc)
options.hiddensize2=20;%100 500
options.hiddensize3=10;%100 500
options.maxepoch1=20;
options.maxepoch2=10;
options.maxepoch3=100;%classification layer
options.maxepoch4=5;
options.trainAll=1;
%try larger flasepositiveratio


sentenceOptions.featureExtractionMethod=@autoEncodeSentences;
sentenceOptions.featureExtractionName='autoEncodeSentences';
sentenceOptions.trainLevel=3; %autoEncode3 autoencoder level
sentenceOptions.sentenceSplit='pairs';
sentenceOptions.hiddensize1=100;%200,100,150,175,100,50
sentenceOptions.hiddensize2=40;%100,50,100,150,25,25
sentenceOptions.maxepoch1=20;
sentenceOptions.maxepoch2=10;
sentenceOptions.maxepoch3=100;
sentenceOptions.sentenceTrainSplit=2000; %no.sentences used to train system
sentenceOptions.force=false;
sentenceOptions.preciseId=false;
 %If precise only match with exact same sentences, important when training image-sentnece association
%passed to sentences to determine how sentenceIds loaded are formatted
%preciseIds means when the representation of sentences is learned only
%by correlating exact matches, then retrieve all sentences representations
%using this trained network
%when options.preciseId only correlate images with exact match sentences
%calculate correlation using this
%images matching always need to be general as no double examples


%% What to run?
matchForce=true;
featureForce=false;
sentenceForce=true;
classifyImages=false;
classifySentenceImages=false;
classifySentences=true;



%% Feature Extractors and Classifiers
%%Features
LOMO_F=1;
ALEX_F=2;
VGG_F=3;
AUTOENCODEIMG_F=4;
AUTOENCODEIMG2_F=5;
%%Classifiers
XQDA_F=1;
TWOCHANNEL_F=2;
TWOCHANNEL2_F=3;
AUTOENCODEMATCHES_F=4;
AUTOENCODEMATCHES3_F=5;
AUTOENCODEMATCHES1_F=6;
FEEDFORWARD_F=7;
TWOCHANNEL3_F=8;
%%Which feature extractors to run
%%Which classifiers to run
featureExtractors= [{LOMO_F, @LOMO};{ALEX_F, @ALEX};{VGG_F, @VGG};{AUTOENCODEIMG_F,@autoEncodeImages};{AUTOENCODEIMG2_F,@autoEncodeImages2d}];%%,{MACH, @MACH}
featureImgDimensions=[128,48; 227,227; 224,224; 128, 48; 50, 50]; %100 40
featureName={'LOMO', 'ALEX', 'VGG', 'autoEncode', 'autoEncode2d'};
imgType={'Std','Ctrl','All','Distort'};

%Used for running multiple sentence extraction methods
AUTOENCODE_F=1;
sentenceExtractions=[{AUTOENCODE_F, @autoEncodeSentences}];
sentenceName={'AUTOENCODER'};
sentenceFeatureRun={AUTOENCODE_F};

%Used for sentence input type
%mean, mode, matrix
%Sentences compared need to be of same mode, norm, size, otherwise they
%will have different vector lengths
sentencesRun={
'mode2_norm3outvectors_phrase_win7_threshold150_size400.txt'
};

sentencesRunType=3; %very important to clarify the kind of sentences we want to be loading (can only hold one type in array)

featureExtractorsRun=[AUTOENCODEIMG2_F];%LOMO_F AUTOENCODEIMG2_F
classifiers= [{XQDA_F, @XQDARUN};{TWOCHANNEL_F, @twoChannel};{TWOCHANNEL2_F, @twoChannel2};{AUTOENCODEMATCHES_F, @autoEncodeMatches};{AUTOENCODEMATCHES3_F, @autoEncodeMatches3};{AUTOENCODEMATCHES1_F, @autoEncodeMatches1}; {FEEDFORWARD_F,@feedForwardMatch};{TWOCHANNEL3_F,@twoChannel3}];
classifiersRun=[AUTOENCODEMATCHES3_F];
sentenceClassifiersRun=[XQDA_F];
imageClassifiersRun=[XQDA_F];
classifierName={'XQDA','twoChannel','twoChannel2', 'autoEncodeMatches','autoEncodeMatches3', 'autoEncodeMatches1', 'feedForward', 'twoChannel3'};
%dimensionMatchMethod='pca'; %pca, first 

features=[];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Import all images
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
imgList = dir([imgDir, '*.png']);%[imgDir, '*.png']
n = length(imgList);

%% Allocate memory
info = imfinfo([imgDir, imgList(1).name]);
imgWidth=40;
imgHeight=100;


%% read categories
person_ids=zeros(n,1);
precisePersonIds=zeros(n,1);
for i=1:n
   name=imgList(i).name;%Format image=06_set=3_id=0001
   temp= strsplit(name,{'image=','_set=','_id=','.png'});
   person_ids(i)= str2double(strcat(temp(3),temp(4)));%str2double()
   precisePersonIds(i)= str2double(strcat(temp(2),temp(3),temp(4)));%str2double()
   %person_ids_char(i)= char(strcat(temp(3),temp(4)));
end
 %% Convert to hotcoding representation LATER WHEN LAST NEEDED AS EASIER TO SORT (better for regression)
%options.noImages=n;
%HOTCODING only comes into play for neural networks/autoencoding ,
%otherwise more convenient to keep with human understandable labels
%Only autoencode3 , image transfer learning for alexnet and vggnet

images = zeros(imgHeight,imgWidth, 3, n, 'uint8');
%% read images
for i = 1 : n
    temp = imread([imgDir, imgList(i).name]);
    images(:,:,:,i) = imresize(temp,[imgHeight imgWidth]);   
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Extract features using all feature extraction methods
%%Store in data directory
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%if feature set doesn't already exist

%%Perform feature extraction
%%Save to data folder
if(classifyImages | classifySentenceImages)
    disp('Checking if features to Extract already exist or forced')
    for i=1:length(featureExtractorsRun)
        %Check if features already exist 
        featureList=dir([strcat(featuresDir,'images/') '*.mat']);
        featuresAvail=[featureList.name];
        currFeatureName=cell2mat(featureName(featureExtractorsRun(i)));
        config=sprintf('_%d_%d_%d',imageOptions.imResizeMethod,imageOptions.imageTrainSplit, imageOptions.noImages);
        config=strjoin(cellfun(@num2str,struct2cell(imageOptions),'UniformOutput',0),'_');
        
        currFeatureName=strrep(strcat(currFeatureName,config,'.mat'),' ', '');
        %Run feature extraction function
        %If being forced or features Available doesnt already exist
        if(featureForce || isequal(strfind(featuresAvail,currFeatureName),[]))
            idx=find(cell2mat(featureExtractors(:,1))==featureExtractorsRun(i),1);
            %Could do error checking here to test match exists: 1x0
            %featureID= cell2mat(featureExtractors(u,1));
            featureFunct= cell2mat(featureExtractors(idx,2));
            fprintf('Extracting current feature %s, place in data directory \n',currFeatureName)

            %Load images according to size of feature extraction process
            if(featureExtractorsRun(i)~=AUTOENCODEIMG_F && featureExtractorsRun(i)~=AUTOENCODEIMG2_F)
                imgWidth=featureImgDimensions(featureExtractorsRun(i),2);
                imgHeight=featureImgDimensions(featureExtractorsRun(i),1);
            else
                imgWidth=imageOptions.width;
                imgHeight=imageOptions.height;
            end
            %images = zeros(imgHeight,imgWidth, 3, n, 'uint8');
            images = readInImages(imgDir, imgList, imgWidth, imgHeight, imageOptions.imResizeMethod, imageOptions);


            %RandonPerm depending on noImages, need to keep associated
            %order of personIds or worthless
            [personIds, features]=featureFunct(images,person_ids, imageOptions); %(:,:,:,1:options.noImages) done inside function 
            
            save(char(strcat(featuresDir,'images/',currFeatureName)),'features', 'personIds');

        else
          fprintf('Already exists. Not extracting current feature %s, config %d %d\n',currFeatureName,imageOptions.imResizeMethod,imageOptions.imageTrainSplit)
        end
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Extract all sentences listed in sentencesRun
%%Store in sentences
%%Create sentence probes and galleries for classification
%%DONT KNOW HOW TO DEAL WITH 3 OCCURENCES OF SENTENCE WITH SAME ID, 
%%Need to have a match and randomly selected with xqda,   but sentence ids
%%not lined up, could do it in xqda, by using find, to add 1 for 1 but lets remove here for now
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if(classifySentenceImages | classifySentences)
    fprintf('Loading sentences and their associated imageIds into matrices \n');
    [sentenceNames,sentences, sentenceIds, resultSentences]= extractDescriptions(sentencesDir, sentencesRun, sentencesRunType, sentenceOptions);
    


        %% Order sentences
        [sentenceIds,idx]=sort(sentenceIds);
        sentences=sentences(:,idx,:);%all the files, sentences,words, word vectors

        %% Remove sentences that dont occur twice
        fprintf('\n Input sentences %d with their associated sentenceIds %d \n', size(sentences,2),size(sentenceIds,1));
        occur=0;
        indexes=[];
        idx=1;
        old=0;
        for i=1:length(sentenceIds)
            if(sentenceIds(i)~=old && occur>1)
                for p=1:2%occur
                   indexes(idx)=i-p;
                   idx=idx+1;
                end      
                occur=1;
                old=sentenceIds(i);
            else
                if(sentenceIds(i)==old)
                    occur=occur+1;
                else
                   old=sentenceIds(i);
                   occur=1;
                end
            end
        end
        sentences2=sentences(:,indexes,:);
        sentenceIds2=sentenceIds(indexes);
        fprintf('\n After removing unique, sentences %d with their associated sentenceIds %d \n', size(sentences,2),size(sentenceIds,1));

        %% Create sentence galleries and labels
    for st=1:size(sentences2,1) 
        sentenceGalFea(st,:,:) = sentences2(st,1:2:end, :);
        sentenceProbFea(st,:,:) = sentences2(st,2:2:end, :);
        sentenceClassLabelGal(st,:)=sentenceIds2(1:2:end);
        sentenceClassLabelProb(st,:)=sentenceIds2(2:2:end);
    end        
    %Does not allowing there to be multiple matches within a same sentence so
    %only applies to first image version create a lack of application?
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Create sentence-->feature projections 
%%Store in descriptions matrix with accompanying descriptions labels
%%matrices
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%






%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Create Probe and Image Galleries for all imageFeatures and sentenceImageFeatures
% galFea and probFea for image-image
% sentenceGalFea and sentenceProbFea for image-sentence classification
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

featureList = dir(strcat(featuresDir,'images/',featureExts));
featuresAvail=[featureList.name];
%%For every image feature set
for i=1:length(featureExtractorsRun)
    currFeatureName=cell2mat(featureName(featureExtractorsRun(i)));
    %config=sprintf('_%d_%d_%d',imageOptions.imResizeMethod,imageOptions.imageTrainSplit, imageOptions.noImages);
    config=strjoin(cellfun(@num2str,struct2cell(imageOptions),'UniformOutput',0),'_');
    currFeatureName=strrep(strcat(currFeatureName,config,'.mat'),' ', '');
    fprintf('Now trying to load %s to perform image feature arrangement into gal/prob\n', currFeatureName);
    %% If feature file exists 
    if(~isequal(strfind(featuresAvail,currFeatureName),[]))
        
        %% Load image feature variables from file  
        fprintf('Currently loading features %s into matrices \n',currFeatureName);
        load(char(strcat(featuresDir,'images/',currFeatureName)));%originally saved as features, personIds
        size(features,1)
        length(imgList)
        if(size(features,1) ~= length(imgList))
            descriptors=features.';
        else
            descriptors=features;
        end
        
        %% Order image features
        [personIds,idx] = sort(personIds);
        descriptors=descriptors(idx,:);  
        %%Group sentences with their associated images into matrices 
        % Sentences are config, sentences, vector representation
        
        %% Remove sentences whose ids have no image match
        %Only do if sentences mode
        if(classifySentenceImages | classifySentences)
            matches=ismember(sentenceIds,personIds); %returns logical 1 where sentenceid appears in presonIds
            fprintf('\n Input sentences %d with their associated sentenceIds %d \n', size(sentences,2),size(sentenceIds,1));
            sentenceIds=sentenceIds(find(matches));
            sentences=sentences(:,find(matches),:);

            %% Remove all images with no sentence match
            %This is done later via, sentence id match organisation but will make things clearer
            matches=ismember(personIds,sentenceIds);
            fprintf('Input images %d with their associated personIds %d \n', size(descriptors,1),size(personIds,1));
            personIds2=personIds(find(matches));
            descriptors2=descriptors(find(matches),:);        

            %Now have sentences with their associated sentenceIds
            %Have descriptors with their associated personIds
            fprintf('Post sentence deletion Now have sentences %d with their associated sentenceIds %d \n', size(sentences,2),size(sentenceIds,1))
            fprintf('Post image deletion Now have descriptors %d with their associated personIds %d \n', size(descriptors2,1),size(personIds2,1))
            fprintf('There are 2478 sentences and 2718 images originally \n')

            %% Create sentenceImages that has all image features in same order as sentences
            % Place both images that match single sentence descriptor id
            if(classifySentenceImages)
                sentenceImages=zeros(size(sentences,1),size(sentences,2)*2, size(descriptors2,2));
                imageIds=zeros(size(sentences,2)*2,1);
                for s= 1:size(sentences,2)
                    sId=sentenceIds(s);%sentence id need to find match in descriptors
                    %for every sentence config
                    for c=1:size(sentences,1)
                        %ids of imageMatches, bit pointless
                        %temp=personIds2(find(personIds2==sId),:);

                        imagesMatch=descriptors2(find(personIds2==sId),:);%2*26960, get indedexes images in descriptors with same id
                        if(size(imagesMatch,1)~=2 && ~sentenceOptions.preciseId)
                            fprintf('Error there are not two image matches for every sentenceId with generalId \n')
                        end
                        imageIds(s)=sId;%temp(1);
                        sentenceImages(c,s,:)= squeeze(imagesMatch(1,:)); %for each sentences there are two images
                        %if(~preciseId)
                            imageIds(s+size(sentences,2))=sId;%temp(2);                        
                            sentenceImages(c,s+size(sentences,2),:)= squeeze(imagesMatch(2,:)); 
                        %end
                    end
                end

                %images are placed with matching id in begin and end in
                %sentenceImages ids are repeated, sentences are similatly repeated
                sentenceImgGalFea(i,:,:,:)=[sentences(:,:,:),sentences(:,:,:)];
                sentenceImgProbFea(i,:,:,:)=sentenceImages(:,:,:);
                sentenceImgClassLabel(i,:)=[sentenceIds(:);sentenceIds(:)];
                fprintf('Final sizes of sentences gallery %d, associated images gallery %d, and their matching sentenceClassLabels %d \n\n', size(sentenceImgGalFea,3), size(sentenceImgProbFea,3), size(sentenceImgClassLabel,2))
            end
        end
        %% Load image feature pairs into galFea, probFea, with associated class labels

        % Sort person Ids and images so can be split effectively for
        % matching
        %now both sorted ascending
        numImages= int16(size(descriptors,1)/2);
        personIds=[personIds(1:2:end);personIds(2:2:end)];
        descriptors=[descriptors(1:2:end,:); descriptors(2:2:end,:)];
        
        galFea(i,:,:) = descriptors(1 : numImages, :);
        probFea(i,:,:) = descriptors(numImages + 1 : end, :);
        classLabelGal(i,:)=personIds(1:numImages);
        classLabelProb(i,:)=personIds(numImages+1:end);
   
    else
        fprintf('Could not load features %s into matrices as folder didnt exist \n',currFeatureName);
    end
end




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Get results for image-sentence ranks
%%Perform classification on all images to respective sentences
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if(classifySentenceImages)
    if(options.testSize~=0 && classifiersRun(i)~=XQDA_F)
        numRanks=options.testSize;
    else
        numRanks=size(sentenceImgGalFea,3)/2-1;
    end
   fprintf('------------------------------------------ \n CLASSIFY SENTENCE IMAGES \n --------------------------------------------\n');
    %numRanks=size(sentenceImgGalFea,3)/2-1;
    cms = zeros(numFolds, numRanks); %only need results within classification within features
    %matchingConfig=sprintf('fpr_%ddmm_%dts_%d',options.falsePositiveRatio,options.dimensionMatchMethod, options.testSize);
    matchingConfig=strjoin(cellfun(@num2str,struct2cell(options),'UniformOutput',0),'');
    labels=cell(length(classifiersRun)*size(sentenceImgGalFea,1)*size(sentenceImgGalFea,2),1);   
    %%Select classifiers want to run
    for i=1:length(classifiersRun)
                idx=find(cell2mat(classifiers(:,1))==classifiersRun(i),1);
                currClassifierId=cell2mat(classifiers(idx,1));
                currClassifierFunct=cell2mat(classifiers(idx,2));
                currClassifierName=cell2mat(classifierName(currClassifierId));
                fprintf('Currently running classifier %s \n',currClassifierName)
                
                %Adjust sentence and images dimensions/projections
                %depending on dimensionMatchmethod and classification method
                fprintf('Adjusting image and sentence feature dimensions so they match %s \n',options.dimensionMatchMethod)
                [sentenceImgGalFea, sentenceImgProbFea]=matchDimensions(sentenceImgProbFea,sentenceImgGalFea, options.dimensionMatchMethod, currClassifierName);
                fprintf('Dimension Matching Completed \n')
                %%For every set of features
                for ft=1:size(sentenceImgGalFea,1)
                    figure

                    %%For every sentence configuration set
                    for st=1:size(sentenceImgGalFea,2)

                        currFeatureName=cell2mat(featureName(featureExtractorsRun(ft)));
                        temp=strrep(resultSentences(st),'../results/sentences/','');
                        config='';%strjoin(cellfun(@num2str,struct2cell(imageOptions),'UniformOutput',0),'-');
                        csvFileName=strcat(resultsDir,'sentenceImages/',currClassifierName,'_',currFeatureName,'_',config,matchingConfig, temp,'.csv');
                        %strcat(currClassifierName,'_',currFeatureName,config,matchingConfig, temp)
                        labels{((i-1)*size(sentenceImgGalFea,1)*size(sentenceImgGalFea,2))+ (size(sentenceImgGalFea,2)*(ft-1))+st}=char(strcat(currClassifierName,'-',currFeatureName,'-',config,matchingConfig, temp));
                        
                        %if there exists no results for this sentence
                        %config, images extracted
                        if (exist(char(strrep(csvFileName,'.csv','.mat')), 'file') ~= 2 || matchForce==true)
                            %Repeat classification process numFolds times
                            for iter=1:numFolds


                                [dist,classLabelGal2, classLabelProb2]=currClassifierFunct(squeeze(sentenceImgGalFea(ft,st,:,:)), squeeze(sentenceImgProbFea(ft,st,:,:)),squeeze(sentenceImgClassLabel(ft,:)),squeeze(sentenceImgClassLabel(ft,:)),iter,options);
                                size(dist)
                                size(classLabelGal2)
                                size(classLabelProb2)
                                cms(iter,:) = EvalCMC( -dist, classLabelGal2, classLabelProb2, numRanks );
                                clear dist           

                                fprintf(' Rank1,  Rank5, Rank10, Rank15, Rank20\n');
                                fprintf('%5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%\n\n', cms(iter,[1,5,10,15,20]) * 100);

                            end
                            %Mean for every feature set, classifier combination
                            meanCms = mean(cms(:,:));
                            save(char(strrep(csvFileName,'.csv','.mat')), 'meanCms');
                        else
                           load( char(strrep(csvFileName,'.csv','.mat')));
                        end
                        plot(1 : numRanks, meanCms)
                        hold on;

                        %csvFileName=strcat(resultsDir,currClassifierName,'_',currFeatureName,'_', config,'_',dimensionMatchMethod,'_', char(sentenceNames(st)));
                        csvwrite(char(csvFileName),meanCms) 

                        fprintf('The average performance:\n');
                        fprintf(' Rank1,  Rank5, Rank10, Rank15, Rank20,  Rank100\n');
                        fprintf('%5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%\n\n', (meanCms([1,5,10,15,20,100]) * 100));

                    end
                    title(sprintf('CMS Curve for CUHK03 Image and Sentence Matching'))
                    xlabel('No. Ranks of ordered Gallery Images') % x-axis label
                    ylabel('% Gallery Images that contain match within that rank') % y-axis label
                    legend(labels);
                    hold off
                end
    end
     
end





%% Get results for image-image ranks %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%For all extracted features 
%%For all classification techniques
%%Features split between galFea, probFea, 
%%Feature labels split between classLabelGal, classLabelFea
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%Select classifiers want to run
if(classifyImages)
    fprintf('------------------------------------------ \n CLASSIFY IMAGES \n --------------------------------------------\n');
    if(options.testSize~=0 && imageClassifiersRun(i)~=XQDA_F)
        numRanks=options.testSize;
    else
        numRanks=int16(size(galFea,2)/2-1);
    end
    cms = zeros(numFolds, numRanks);
    
    figure

    labels=cell(length(imageClassifiersRun)*size(galFea,1),1);
    for i=1:length(imageClassifiersRun)
            idx=find(cell2mat(classifiers(:,1))==imageClassifiersRun(i),1);
            currClassifierId=cell2mat(classifiers(idx,1));
            currClassifierFunct=cell2mat(classifiers(idx,2));
            currClassifierName=cell2mat(classifierName(currClassifierId));
            fprintf('Currently running classifier %s \n',currClassifierName)
                %%For every set of features
                for ft=1:size(galFea,1)
                    currFeatureName=cell2mat(featureName(featureExtractorsRun(ft)));
                    %config=sprintf('%d-%d-%d',imageOptions.imResizeMethod,imageOptions.imageTrainSplit,imageOptions.noImages);
                    config=strjoin(cellfun(@num2str,struct2cell(imageOptions),'UniformOutput',0),'-');
                    labels{(size(galFea,1)*(i-1))+ft}=char(strcat(currClassifierName,'-',currFeatureName,'-', config));                     

                    csvFileName=strcat(resultsDir,'images/',currClassifierName,'-',currFeatureName,'-', config,'.csv');
                    
                    
                    if (exist(char(strrep(csvFileName,'.csv','.mat')), 'file') ~= 2 || matchForce==true)
                        %Repeat classification process numFolds times
                        for iter=1:numFolds

                            [dist,classLabelGal2, classLabelProb2]=currClassifierFunct(squeeze(galFea(ft,:,:)), squeeze(probFea(ft,:,:)),squeeze(classLabelGal(ft,:)),squeeze(classLabelProb(ft,:)),iter, options);

                            cms(iter,:) = EvalCMC( -dist, classLabelGal2, classLabelProb2, numRanks );
                            clear dist           

                            fprintf(' Rank1,  Rank5, Rank10, Rank15, Rank20\n');
                            fprintf('%5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%\n\n', cms(iter,[1,5,10,15,20]) * 100);

                        end        
                        %Mean for every feature set, classifier combination
                        meanCms = mean(cms(:,:));          
                        save(char(strrep(csvFileName,'.csv','.mat')), 'meanCms');
                    else
                        load( char(strrep(csvFileName,'.csv','.mat')));
                    end                       

                    
                    plot(1 : numRanks, meanCms)
                    hold on;
                    csvwrite(csvFileName,meanCms); 
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
                    %%type csvlist.dat
                    
                    fprintf('The average performance:\n');
                    fprintf(' Rank1,  Rank5, Rank10, Rank15, Rank50\n');
                    fprintf('%5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%\n\n', meanCms([1,5,10,15,50]) * 100);
                end
           % end
       % end
    end
   title(sprintf('CMS Curve for CUHK03 Image Matching'))
    xlabel('No. Ranks of ordered Gallery Images') % x-axis label
    ylabel('% Gallery Images that contain match within that rank') % y-axis label
    legend(labels);
    hold off;
end



%% Get results for sentence-sentence ranks %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%For all sentences 
%%For all classification techniques
%%Features split between sentenceGalFea, sentenceProbFea, 
%%Feature labels split between sentenceClassLabelGal, sentenceClassLabelFea
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%Select classifiers want to run
if(classifySentences)
    fprintf('------------------------------------------ \n CLASSIFY SENTENCES \n --------------------------------------------\n');
    if(options.testSize~=0 && sentenceClassifiersRun(i)~=XQDA_F)
        numRanks=options.testSize;
    else
        numRanks=int16(size(sentenceGalFea,2)/2-1);
    end
    
    cms = zeros(numFolds, numRanks);
   
    figure    

    labels=cell(length(sentenceClassifiersRun)*size(sentenceGalFea,1),1);
    
    for i=1:length(sentenceClassifiersRun)
            idx=find(cell2mat(classifiers(:,1))==sentenceClassifiersRun(i),1);
            currClassifierId=cell2mat(classifiers(idx,1));
            currClassifierFunct=cell2mat(classifiers(idx,2));
            currClassifierName=cell2mat(classifierName(currClassifierId));
            fprintf('Currently running classifier %s \n',currClassifierName)
                %%For every sentence configuration set
                for st=1:size(sentenceGalFea,1)
                    %Repeat classification process numFolds times
                    temp=strrep(resultSentences(st),'../results/sentences/','');
                    csvFileName=char(strcat('../results/sentences/',currClassifierName,'_', temp));                   

                    labels{(size(sentenceGalFea,1)*(i-1))+st}=strrep(csvFileName,'.csv','');
                    if (exist(char(strrep(csvFileName,'.csv','.mat')), 'file') ~= 2 || matchForce==true)

                        for iter=1:numFolds

                            [dist,classLabelGal2, classLabelProb2]=currClassifierFunct(squeeze(sentenceGalFea(st,:,:)), squeeze(sentenceProbFea(st,:,:)),squeeze(sentenceClassLabelGal(st,:)),squeeze(sentenceClassLabelProb(st,:)),iter, options);

                            cms(iter,:) = EvalCMC( -dist, classLabelGal2, classLabelProb2, numRanks );
                            clear dist           

                            fprintf(' Rank1,  Rank10, Rank20, Rank50\n');
                            fprintf('%5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%\n\n', cms(iter,[1,10,20,50]) * 100);

                        end
                        %Mean for every feature set, classifier combination
                        meanCms = mean(cms(:,:));
                        save(char(strrep(csvFileName,'.csv','.mat')), 'meanCms');
                    else
                        load( char(strrep(csvFileName,'.csv','.mat')));
                    end   
                    
                   
                    plot(1 : numRanks, meanCms)
                    hold on;
                    

                    csvwrite(csvFileName,meanCms)   

                    fprintf('The average performance:\n');
                    fprintf(' Rank1,  Rank10, Rank20, Rank50, Rnk60, Rank70, Rank80, Rank90\n');
                    fprintf('%5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%,%5.2f%%\n\n', meanCms([1,10,20,50,60,70,80,90]) * 100);
                end
           % end
       % end
    end
    title(sprintf('CMS Curve for sentence matches'))
    xlabel('No. Ranks of ordered Gallery Images') % x-axis label
    ylabel('% Gallery Images that contain match within that rank') % y-axis label
    legend(labels);
    hold off;
end






