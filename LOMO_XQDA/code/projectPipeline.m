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
sentencesDir='../../word2vec/trunk/matlab_sentence_vectors/';

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
numRanks = 100; %Number of ranks to show for
READ_STD=1;
READ_CENTRAL=2;
READ_ALL=3;
options.imResizeMethod=READ_ALL;
options.trainSplit=0.6;
options.sentenceSplit='pairs'; %'oneofeach' 'oneofeach+' 
options.noImages=0;%if 0 then all run
options.featureExtractionMethod='AUTOENCODE3';%AUTOENCODE2, LOMO

%% What to run?
featureForce=false; 
classifyImages=true;
classifySentenceImages=false;
classifySentences=false;

%% Feature Extractors and Classifiers
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



sentencesRun={'mode0_norm3outvectors_phrase_win3_threshold100_size50.txt'}; %'all' leads to running every sentence vector
sentencesRunType=3;

featureExtractorsRun=[ALEX_F];%LOMO_F
classifiers= [{XQDA_F, @XQDARUN}];
classifiersRun=[XQDA_F];
classifierName={'XQDA'};
dimensionMatchMethod='first'; %pca, first 
generaliseMatching=false; %If true every sentence is matched to both images that match its id

preciseId=false; %If precise only match with exact same sentences

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


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Extract features using all feature extraction methods
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

        %Load images according to size of feature extraction process
        imgWidth=featureImgDimensions(featureExtractorsRun(i),2);
        imgHeight=featureImgDimensions(featureExtractorsRun(i),1);
        %images = zeros(imgHeight,imgWidth, 3, n, 'uint8');
        images = readInImages(imgDir, imgList, imgWidth, imgHeight, options.imResizeMethod);


        %RandonPerm depending on noImages, need to keep associated
        %order of personIds or worthless
        [personIds, features]=featureFunct(images,person_ids, options); %(:,:,:,1:options.noImages) done inside function 

        save(char(strcat(featuresDir,currFeatureName)),'features', 'personIds');

    else
      fprintf('Already exists. Not extracting current feature %s, config %d %d %d \n',currFeatureName,options.imResizeMethod,options.trainSplit,options.noImages)
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
    [sentenceNames,sentences, sentenceIds]= extractDescriptions(sentencesDir, sentencesRun, preciseId, sentencesRunType, options);



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

featureList = dir(strcat(featuresDir,featureExts));
featuresAvail=[featureList.name];
%%For every image feature set
for i=1:length(featureExtractorsRun)
    currFeatureName=cell2mat(featureName(featureExtractorsRun(i)));
    config=sprintf('%d_%d_%d_',options.imResizeMethod,int16(options.trainSplit*100),options.noImages);
    currFeatureName=strrep(strcat(config,currFeatureName),' ', '');
    
    %% If feature file exists 
    if(~isequal(strfind(featuresAvail,currFeatureName),[]))
        
        %% Load image feature variables from file  
        fprintf('Currently loading features %s into matrices \n',currFeatureName);
        load(char(strcat(featuresDir,currFeatureName)));%originally saved as features, personIds
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
            sentenceImages=zeros(size(sentences,1),size(sentences,2)*2, size(descriptors2,2));
            imageIds=zeros(size(sentences,2)*2,1);
            for s= 1:size(sentences,2)
                sId=sentenceIds(s);%sentence id need to find match in descriptors
                for c=1:size(sentences,1)
                    temp=personIds2(find(personIds2==sId),:);

                    imagesMatch=descriptors2(find(personIds2==sId),:);%2*26960, get indedexes images in descriptors with same id
                    if(size(imagesMatch,1)~=2)
                        fprintf('Error there are not two image matches for every sentenceId \n')
                    end
                    imageIds(s)=temp(1);
                    imageIds(s+size(sentences,2))=temp(2);
                    sentenceImages(c,s,:)= squeeze(imagesMatch(1,:)); %for each sentences there are two images
                    sentenceImages(c,s+size(sentences,2),:)= squeeze(imagesMatch(2,:)); 
                end
            end

            %images are placed with matching id in begin and end in
            %sentenceImages ids are repeated, sentences are similatly repeated
            sentenceImgGalFea(i,:,:,:)=[sentences(:,:,:),sentences(:,:,:)];
            sentenceImgProbFea(i,:,:,:)=sentenceImages(:,:,:);
            sentenceImgClassLabel(i,:)=[sentenceIds(:);sentenceIds(:)];
            fprintf('Final sizes of sentences gallery %d, associated images gallery %d, and their matching sentenceClassLabels %d \n\n', size(sentenceImgGalFea,3), size(sentenceImgProbFea,3), size(sentenceImgClassLabel,2))
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
    numRanks=size(sentenceImgGalFea,3)/2-1;
    cms = zeros(numFolds, numRanks); %only need results within classification within features

    %%Select classifiers want to run
    for i=1:length(classifiersRun)
                idx=find(cell2mat(classifiers(:,1))==classifiersRun(i),1);
                currClassifierId=cell2mat(classifiers(idx,1));
                currClassifierFunct=cell2mat(classifiers(idx,2));
                currClassifierName=cell2mat(classifierName(currClassifierId));
                fprintf('Currently running classifier %s \n',currClassifierName)
                
                %Adjust sentence and images dimensions/projections
                %depending on dimensionMatchmethod and classification method
                [sentenceImgGalFea, sentenceImgProbFea]=matchDimensions(sentenceImgProbFea,sentenceImgGalFea, dimensionMatchMethod, currClassifierName);

                %%For every set of features
                for ft=1:size(sentenceImgGalFea,1)
                    figure

                    %%For every sentence configuration set
                    for st=1:size(sentenceImgGalFea,2)

                        %Repeat classification process numFolds times
                        for iter=1:numFolds


                           [dist,classLabelGal2, classLabelProb2]=currClassifierFunct(squeeze(sentenceImgGalFea(ft,st,:,:)), squeeze(sentenceImgProbFea(ft,st,:,:)),squeeze(sentenceImgClassLabel(ft,:)),squeeze(sentenceImgClassLabel(ft,:)),iter);

                            cms(iter,:) = EvalCMC( -dist, classLabelGal2, classLabelProb2, numRanks );
                            clear dist           

                            fprintf(' Rank1,  Rank5, Rank10, Rank15, Rank20\n');
                            fprintf('%5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%\n\n', cms(iter,[1,5,10,15,20]) * 100);

                        end
                        %Mean for every feature set, classifier combination
                        meanCms = mean(cms(:,:));

                        plot(1 : numRanks, meanCms)

                        legend(char(sentenceNames(:)));
                        hold on;
                        currFeatureName=cell2mat(featureName(featureExtractorsRun(ft)));
                        config=sprintf('%d_%d_%d',options.imResizeMethod,int16(options.trainSplit*100),options.noImages);

                        csvFileName=strcat(resultsDir,currClassifierName,'_',currFeatureName,'_', config,'_',dimensionMatchMethod,'_', char(sentenceNames(st)));
                        csvwrite(strrep(csvFileName,'.txt','.csv'),meanCms)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
                        %%type csvlist.dat

                        fprintf('The average performance:\n');
                        fprintf(' Rank1,  Rank5, Rank10, Rank15, Rank20,  Rank100, Rank500, Rank1000\n');
                        fprintf('%5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%,   %5.2f%%,   %5.2f%%\n\n', (meanCms([1,5,10,15,20,100,500,1000]) * 100));

                    end
                    title(sprintf('CMS Curve for Classifier %s, feature set %s,settings %s, dimension reduce method ', currClassifierName, currFeatureName, config, dimensionMatchMethod))
                    xlabel('No. Ranks of ordered Gallery Images') % x-axis label
                    ylabel('% Sentences that contain match within that rank') % y-axis label
                end
       % end
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
    numRanks=int16(size(galFea,2)/2-1);
    cms = zeros(numFolds, numRanks);
    
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

                        [dist,classLabelGal2, classLabelProb2]=currClassifierFunct(squeeze(galFea(ft,:,:)), squeeze(probFea(ft,:,:)),squeeze(classLabelGal(ft,:)),squeeze(classLabelProb(ft,:)),iter);
                      
                        cms(iter,:) = EvalCMC( -dist, classLabelGal2, classLabelProb2, numRanks );
                        clear dist           

                        fprintf(' Rank1,  Rank5, Rank10, Rank15, Rank20\n');
                        fprintf('%5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%\n\n', cms(iter,[1,5,10,15,20]) * 100);

                    end
                    %Mean for every feature set, classifier combination
                    meanCms = mean(cms(:,:));
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
                    fprintf(' Rank1,  Rank5, Rank10, Rank15, Rank50\n');
                    fprintf('%5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%\n\n', meanCms([1,5,10,15,50]) * 100);
                end
           % end
       % end
    end
end



%% Get results for sentence-sentence ranks %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%For all sentences 
%%For all classification techniques
%%Features split between sentenceGalFea, sentenceProbFea, 
%%Feature labels split between sentenceClassLabelGal, sentenceClassLabelFea
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%Select classifiers want to run
if(classifySentences)
    numRanks=int16(size(sentenceGalFea,2)/2-1);
    cms = zeros(numFolds, numRanks);
    
    for i=1:length(classifiersRun)
            idx=find(cell2mat(classifiers(:,1))==classifiersRun(i),1);
            currClassifierId=cell2mat(classifiers(idx,1));
            currClassifierFunct=cell2mat(classifiers(idx,2));
            currClassifierName=cell2mat(classifierName(currClassifierId));
            fprintf('Currently running classifier %s \n',currClassifierName)
                %%For every sentence configuration set
                for st=1:size(sentenceGalFea,1)
                    %Repeat classification process numFolds times
                    for iter=1:numFolds

                        [dist,classLabelGal2, classLabelProb2]=currClassifierFunct(squeeze(sentenceGalFea(st,:,:)), squeeze(sentenceProbFea(st,:,:)),squeeze(sentenceClassLabelGal(st,:)),squeeze(sentenceClassLabelProb(st,:)),iter);
                      
                        cms(iter,:) = EvalCMC( -dist, classLabelGal2, classLabelProb2, numRanks );
                        clear dist           

                        fprintf(' Rank1,  Rank5, Rank10, Rank15\n');
                        fprintf('%5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%\n\n', cms(iter,[1,5,10,15]) * 100);

                    end
                    %Mean for every feature set, classifier combination
                    meanCms = mean(cms(:,:));
                    figure
                    plot(1 : numRanks, meanCms)
                    
                    title(sprintf('CMS Curve for Classifier %s,sentences %s', currClassifierName, char(sentenceNames(st))))
                    xlabel('No. Ranks of ordered Gallery Images') % x-axis label
                    ylabel('% Gallery Images that contain match within that rank') % y-axis label

                    csvFileName=strcat(resultsDir,currClassifierName,'_',char(sentenceNames(st)),'_', config);
                    csvwrite(csvFileName,meanCms)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   
                    %%type csvlist.dat

                    fprintf('The average performance:\n');
                    fprintf(' Rank1,  Rank5, Rank10, Rank15\n');
                    fprintf('%5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%\n\n', meanCms([1,5,10,15]) * 100);
                end
           % end
       % end
    end
end








