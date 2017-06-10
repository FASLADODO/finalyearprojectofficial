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
function [dist,classLabelGal2, classLabelProb2]=autoEncodeMatches(galFea, probFea,galClassLabel,probClassLabel, iter, options)
    threshAim=1;
    
    %% Assert that gal and probe labels match (they should)
    'WHether the gal and prob, sentences and images are matching labels'
    isequal(galClassLabel,probClassLabel)
    
    fprintf('generating galFea1, probFea1, with correct match labels, in proportion to falsePositiveRatio\n')
    %% Create training positive matches
    numExamples=size(galFea,1);
    p = randperm(numExamples);
    galFea1 = galFea(p(1:int16(numExamples/2)), : );
    probFea1 = probFea(p(1:int16(numExamples/2)), : );
    numMatches=size(galFea1,1);
    classLabelGal1=galClassLabel(p(1:numMatches));
    classLabelProb1=probClassLabel(p(1:numMatches));
    
    matchResults=ones(numMatches,1)*threshAim;
    temp=zeros((numMatches*options.falsePositiveRatio),1);
    size(temp);
    
    matchResults((1+numMatches):(((1+options.falsePositiveRatio)*numMatches)),1)=temp;
    
    
    %% Now add training false matches
    for i= 1: numMatches
        falseMatchIndexes=find(classLabelProb1 ~= classLabelGal1(i));
        n = randperm(length(falseMatchIndexes));
        for f=1:options.falsePositiveRatio
            
            galFea1(i+numMatches*f,:)=galFea1(i,:);
            probFea1(i+numMatches*f,:)= probFea1(falseMatchIndexes(n(f)),:);%Take random negative match  
        end
    end
    %can go through pairs by first index, creates two columns, each col
    %represents an image/sentence feature set
    %trainingPairs= cat(3,galFea1,probFea1);
    size(galFea1); %should be no.examples, no.features
    size(probFea1);
    half=(size(probFea,2)/2);
    fprintf('Assembling joint input for trainingNetwork2');
    for i=1:numMatches*(options.falsePositiveRatio+1)
        size(galFea1(i,:));
        size(probFea1(i,:));
        trainingPairs(:,1,i)=galFea1(i,:);
        size(trainingPairs);
        trainingPairs(:,2,i)=probFea1(i,:);
        size(trainingPairs);
    end
    
    %% Zero mean. unit variance training
    for i=1:size(trainingPairs,3)
        trainingPairs(:,:,i)=(trainingPairs(:,:,i)-mean2(trainingPairs(:,:,i)))/std2(trainingPairs(:,:,i));
    end
    %% Encoder parameters
    rng('default');%explicit set random seed, so results replicable
    hiddenSize1 = options.hiddensize1;%size of hidden layer in autoencoder, want smaller than sentences
    hiddenSize2= options.hiddensize2;%int16(0.5*size(sentences,4))
    
                pairsTrainIn=cell(size(trainingPairs,3),1);
                 for i=1:size(trainingPairs,3)
                   pairsTrainIn{i}=squeeze(trainingPairs(:,:,i));
                 end


                fprintf('Running autoencoder1...\n');
                autoenc1 = trainAutoencoder(pairsTrainIn,hiddenSize1, ...
                'MaxEpochs',options.maxepoch1, ...%200
                'L2WeightRegularization',0.004, ... %impact of L2 reglarizer on network weights
                'SparsityRegularization',4, ... %impact sparcity regularizer, constrains sparsity of hidden layer output
                'SparsityProportion',0.15, ...%each hidden layer neuron proportion that output
                'ScaleData', false); 
                view(autoenc1)
                fprintf('Running autoencoder2...\n');
                features1=encode(autoenc1, pairsTrainIn);
                autoenc2 = trainAutoencoder(features1,hiddenSize2, ...
                    'MaxEpochs',options.maxepoch2, ...%100
                    'L2WeightRegularization',0.002, ...
                    'SparsityRegularization',4, ...
                    'SparsityProportion',0.1, ...
                    'ScaleData', false);
                features2 = encode(autoenc2,features1);
                'size features2'
                size(features2)
                'size matchresults'
                size(matchResults)
            
                softnet = trainSoftmaxLayer(features2,matchResults.','MaxEpochs',options.maxepoch3);
                deepnet = stack(autoenc1,autoenc2,softnet);

                
                    fprintf(' Training autoencoders with examples...\n');
                    inputSize = size(trainingPairs,1)*size(trainingPairs,2);
                    % Turn the training images into vectors and put them in a matrix
                    'number of xtrain examples'
                    size(trainingPairs,3)
                    xTrain = zeros(inputSize,size(trainingPairs,3));
                    for i = 1:size(trainingPairs,3)
                        xTrain(:,i) = pairsTrainIn{i}(:);
                    end
                    
                    %deepnet.numInputs
                    'size xTrain'
                    size(xTrain)
                    'size xMatchResults'
                    size(matchResults)
                    deepnet = train(deepnet,xTrain,matchResults.');
                    %deepnet.outputConnect=[0, 0,1];
                    
                    %% Need to assemble test all combinations
                    galFea2 = galFea(p(numMatches+1:end), : );
                    probFea2 = probFea(p(numMatches+1:end), : );
                    classLabelGal2=galClassLabel(p(numMatches+1:end));
                    classLabelProb2=probClassLabel(p(numMatches+1:end));
                    
                    %% Reduce size of tests
                    testSize=min(options.testSize, size(galFea2,1));
                    galFea2=galFea2(1:testSize,:);
                    probFea2=probFea2(1:testSize,:);
                    classLabelGal2=classLabelGal2(1:testSize);
                    classLabelProb2=classLabelProb2(1:testSize);
                    
                    fprintf('Creating distance vector...\n');
                    dist=zeros(size(galFea2,1),size(probFea2,1));
                    xAll = zeros(inputSize,1);
                    
                    %% Test every combination and store in results dist matrix
                    for i = 1:size(galFea2,1)
                        fprintf('Currently tested %d/%d\n', i, size(galFea2,1))
                        for u=1:size(probFea2,1)
                            xAll(:,1)=[galFea(i,:),probFea1(u,:)].';
                            xAll=(xAll-mean2(xAll))/std2(xAll);
                            values=deepnet(xAll);
                            match=classLabelGal2(i)==classLabelProb2(u);
                            dist(i,u)=-abs(threshAim-values);  %ones down centre should match
                            if(match && i~=1 && u~=1)
                               fprintf('match %0.2f and nearest wrong neighbour %0.2f \n',dist(i,u), dist(i,u-1)) 
                            end
                        end
                    end
                    

    
end