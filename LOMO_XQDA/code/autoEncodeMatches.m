%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Implements a 2 channel autoEncoder machine learning, producing 1 or 0 for matches
%% Need to create false and positive examples to train, then brute force the examples to produce results.
%dist is 1-x result so varies 0-->1
%GalFea and ProbFea have been reduced to same dimension
%Gallery-sentences, Probe-images
%Traditionally randomly split evenly for train and testing
%Labels are the same

%Produces the match matrix between images and sentences
%Training images are indexed by their last index
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
function [dist,classLabelGal2, classLabelProb2]=twoChannel2(galFea, probFea,galClassLabel,probClassLabel, iter, options)
    testSize=options.testSize;
    
    %% Assert that gal and probe labels match (they should)
    'WHether the gal and prob, sentences and images are matching labels'
    isequal(galClassLabel,probClassLabel)
    
    fprintf('generating galFea1, probFea1, with correct match labels, in proportion to falsePositiveRatio')
    %% Create training positive matches
    numMatches=size(galFea,1);
    p = randperm(numMatches);
    galFea1 = galFea(p(1:int16(numMatches/2)), : );
    probFea1 = probFea(p(1:int16(numMatches/2)), : );
    classLabelGal1=galClassLabel(p(1:int16(numMatches/2)));
    classLabelProb1=probClassLabel(p(1:int16(numMatches/2)));
    matchResults=ones(numMatches,1);
    temp=zeros((int16(numMatches/2)*options.falsePositiveRatio),1);
    matchResults((1+int16(numMatches/2)):((1+options.falsePositiveRatio)*int16(numMatches/2)),1)=temp;
    
    
    %% Now add training false matches
    for i= 1: int16(numMatches/2)
        falseMatchIndexes=find(classLabelProb1 ~= classLabelGal1(i));
        n = randperm(length(falseMatchIndexes));
        for f=1:options.falsePositiveRatio
            diff=options.falsePositiveRatio-1;
            galFea1(i+int16(numMatches/2)*f,:)=galFea(p(i),:);
            probFea1(i+int16(numMatches/2)*f,:)= probFea1(falseMatchIndexes(n(f)),:);%Take random negative match  
        end
    end
    %can go through pairs by first index, creates two columns, each col
    %represents an image/sentence feature set
    %trainingPairs= cat(3,galFea1,probFea1);
    size(galFea1) %should be no.examples, no.features
    size(probFea1)
    half=(size(probFea,2)/2);
    fprintf('Assembling joint input for trainingNetwork2');
    for i=1:int16(numMatches/2)*(options.falsePositiveRatio+1)
        size(galFea1(i,:))
        size(probFea1(i,:))
        trainingPairs(:,1,i)=galFea1(i,:);
        size(trainingPairs)
        trainingPairs(:,2,i)=probFea1(i,:);
        size(trainingPairs)
    end
    
        %% Encoder parameters
    rng('default');%explicit set random seed, so results replicable
    hiddenSize1 = options.hiddensize1;%size of hidden layer in autoencoder, want smaller than sentences
    hiddenSize2= options.hiddensize2;%int16(0.5*size(sentences,4))
    
                imagesTrainIn2=cell(size(imagesTrain2,3),1);

                imagesAllIn=cell(size(imagesProcess,3),1);

                for i=1:size(imagesProcess,3)
                   imagesAllIn{i}=squeeze(imagesProcess(:,:,i));
                end

                %trainingIn and testingIn based off subset
                %options.sentenceTrainSplit
                 for i=1:size(imagesTrain2,3)
                   imagesTrainIn2{i}=squeeze(imagesTrain2(:,:,i));
                 end



                %% Train autoencoder *2 , create deepnet, get classification results
                % do supervised learning
                'size imagesTrainIn2'
                size(imagesTrainIn2)
                'size imagesIdsTrain2'
                size(imagesIdsTrain2)
                fprintf('Running autoencoder1...\n');
                autoenc1 = trainAutoencoder(imagesTrainIn2,hiddenSize1, ...
                'MaxEpochs',options.maxepoch1, ...%200
                'L2WeightRegularization',0.004, ... %impact of L2 reglarizer on network weights
                'SparsityRegularization',4, ... %impact sparcity regularizer, constrains sparsity of hidden layer output
                'SparsityProportion',0.15, ...%each hidden layer neuron proportion that output
                'ScaleData', false); 
                view(autoenc1)
                fprintf('Running autoencoder2...\n');
                features1=encode(autoenc1, imagesTrainIn2);
                autoenc2 = trainAutoencoder(features1,hiddenSize2, ...
                    'MaxEpochs',options.maxepoch2, ...%100
                    'L2WeightRegularization',0.002, ...
                    'SparsityRegularization',4, ...
                    'SparsityProportion',0.1, ...
                    'ScaleData', false);
                features2 = encode(autoenc2,features1);
                'size features2'
                size(features2)
                'size imagesIdsTrain2'
                size(imagesIdsTrain2)
                softnet = trainSoftmaxLayer(features2,imagesIdsTrain2.','MaxEpochs',options.maxepoch3);
                deepnet = stack(autoenc1,autoenc2,softnet);

                
                    fprintf(' Training autoencoders with examples...\n');
                    inputSize = size(imagesTrain2,1)*size(imagesTrain2,2);
                    % Turn the training images into vectors and put them in a matrix
                    'number of xtrain examples'
                    size(imagesTrain2,3)
                    xTrain = zeros(inputSize,size(imagesTrain2,3));
                    
                    
                    for i = 1:size(imagesTrain2,3)
                        xTrain(:,i) = imagesTrainIn2{i}(:);
                    end
                    
                    xAll = zeros(inputSize,size(imagesProcess,3));
                    for i = 1:size(imagesProcess,3)
                        xAll(:,i) = imagesAllIn{i}(:);
                    end

                    %deepnet.numInputs
                    size(xTrain)
                    size(imagesIdsTrain2.')
                    deepnet = train(deepnet,xTrain,imagesIdsTrain2.');
                    deepnet.outputConnect=[0, 1, 0];

                personIds=imagesIdsProcess;

                features2=deepnet(xAll);
    
end