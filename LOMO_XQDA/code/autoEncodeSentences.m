%THIS IS BASIC BY PRODUCING
%Sentences are 4D sentenceConfig, sentence, word, wordvector
%Want to produce 3D sentenceconfig, sentence, sentencevector
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% DEFAULTS
%HIDEENSIZE1 2*SIZE SENTENCES 4
%hiddensize2 size sentences 4
%'MaxEpochs',400, ...
%'L2WeightRegularization',0.004, ... %impact of L2 reglarizer on network weights
%'SparsityRegularization',4, ... %impact sparcity regularizer, constrains sparsity of hidden layer output
%'SparsityProportion',0.15,
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [sentences,sentenceIds]=autoEncodeSentences(sentences, sentenceIds, options)

    %% Encoder parameters
    rng('default')%explicit set random seed, so results replicable
    hiddenSize1 = 4*size(sentences,4);%size of hidden layer in autoencoder, want smaller than sentences
    hiddenSize2=2*size(sentences,4);
    
    
    for config = 1:size(sentences,1)
        %size(squeeze(sentences(config,:,:,:)))
        %size(num2cell(squeeze(sentences(config,:,:,:)),1))
        %size(num2cell(squeeze(sentences(config,:,:,:)),2))
        %size(num2cell(squeeze(sentences(config,:,:,:)),3))
        %IDX(1:size(sentences,2)) = size(sentences,4); 
        %size(mat2cell(squeeze(sentences(config,:,:,:)),[size(sentences,2)], [size(sentences,3)], [size(sentences,4)]))
        %sentencesIn=mat2cell(squeeze(sentences(config,:,:,:)),[size(sentences,2)], [size(sentences,3)], [size(sentences,4)])
        %size(mat2cell(squeeze(sentences(config,:,:,:)), size(sentences,2)))
        %sentencesIn=num2cell(squeeze(sentences(config,:,:,:)),1);
        %sentencesIn=mat2cell(squeeze(sentences(config,:,:,:)), size(sentences,2));
        
        %%  If autoencoder is type 1 or 2 will just unsupervised extract features
        if(strcmp(options.featureExtractionMethod, 'AUTOENCODE2') | strcmp(options.featureExtractionMethod, 'AUTOENCODE1'))

            sentencesIn=cell(size(sentences,2),1);
            for i=1:size(sentences,2)
               sentencesIn{i}=squeeze(sentences(config,i,:,:));
            end
            size(sentencesIn)

            autoenc1 = trainAutoencoder(sentencesIn,hiddenSize1, ...
            'MaxEpochs',400, ...
            'L2WeightRegularization',0.004, ... %impact of L2 reglarizer on network weights
            'SparsityRegularization',4, ... %impact sparcity regularizer, constrains sparsity of hidden layer output
            'SparsityProportion',0.1, ...%each hidden layer neuron proportion that output
            'ScaleData', false); 
            view(autoenc1)
            feat1=encode(autoenc1, sentencesIn);
            fprintf('the size of the features generated is')
            size(feat1)
            sentences2(config,:,:)=feat1.';

            %Take these features to next layer, then reduce down , then
            %backtrain to make features with labelled data more accurate
            %% If AUTOENCODE2 reduce size of feature data further for more intense representation
            if(strcmp(options.featureExtractionMethod, 'AUTOENCODE2'))
                autoenc2 = trainAutoencoder(feat1,hiddenSize2, ...
                    'MaxEpochs',100, ...
                    'L2WeightRegularization',0.002, ...
                    'SparsityRegularization',4, ...
                    'SparsityProportion',0.1, ...
                    'ScaleData', false);
                feat2 = encode(autoenc2,feat1);
                size(feat2)
                sentences3(config,:,:)=feat2.';

            end
        end
        %% If autoencode3 will be extra training deep neural network, will need to split data to train and test
        if(strcmp(options.featureExtractionMethod, 'AUTOENCODE3'))
            %% Order sentences
            [sentenceIdsProcess,idx]=sort(sentenceIds);
            sentencesProcess=squeeze(sentences(config,idx,:,:));%all the files, sentences,words, word vectors
            fprintf('\nthe size of sentencesprocess is\n')
            size(sentencesProcess)
            occur=0;
            indexes=[];
            idx=1;
            old=0;
            %%Construct sentencesTest and sentencesTrain based on settings,
            % Get indexes for train data
            switch options.sentenceSplit
                case 'pairs'
                        fprintf('Training data is all pairs that exist of sentenceData\n');
                        for i=1:length(sentenceIdsProcess)
                            if(sentenceIdsProcess(i)~=old && occur>1)
                                for p=1:2%occur
                                   indexes(idx)=i-p;
                                   idx=idx+1;                               
                                end      
                                occur=1;
                                old=sentenceIdsProcess(i);
                            else
                                if(sentenceIdsProcess(i)==old)
                                    occur=occur+1;
                                else
                                   old=sentenceIdsProcess(i);
                                   occur=1;
                                end
                            end
                        end   
                case 'oneofeach'
                        fprintf('Training data is one of each of all sentence data');
                       for i=1:length(sentenceIdsProcess)
                            if(sentenceIdsProcess(i)~=old && occur>1)
                                
                                indexes(idx)=i-p;
                                idx=idx+1;
                                      
                                occur=1;
                                old=sentenceIdsProcess(i);
                            else
                                if(sentenceIdsProcess(i)==old)
                                    occur=occur+1;
                                else
                                   old=sentenceIdsProcess(i);
                                   occur=1;
                                end
                            end
                        end
                    
                case 'oneofeach+'
                       fprintf('Training data is one of each of all sentence data + extras');
                       for i=1:length(sentenceIdsProcess)
                            if(sentenceIdsProcess(i)~=old && occur>1)
                                
                                indexes(idx)=i-p;
                                idx=idx+1;
                                      
                                occur=1;
                                old=sentenceIdsProcess(i);
                            else
                                if(sentenceIdsProcess(i)==old)
                                    occur=occur+1;
                                else
                                   old=sentenceIdsProcess(i);
                                   occur=1;
                                end
                            end
                        end                   
                        indexes= setdiff([1:size(sentencesProcess,1)],indexes);
            end       
            %create sentencesTrain and sentencesTest
            %sentneceProcess is all current data in this configfile
            sentencesTrain=sentencesProcess(indexes,:,:);
            sentencesIdsTrain=sentenceIdsProcess(indexes);
            testIndexes= setdiff([1:size(sentencesProcess,1)],indexes);
            sentencesTest=sentencesProcess(testIndexes,:,:);
            sentencesIdsTest=sentenceIdsProcess(testIndexes);
            fprintf('the size of sentencestest is\n')
            size(sentencesTest)
            fprintf('the size of sentencestrain is\n')            
            size(sentencesTrain)
            
            sentencesTrainIn=cell(size(sentencesTrain,1),1);
            sentencesTestIn=cell(size(sentencesTest,1),1);
            fprintf('the size of sentencestrain is\n') 
            size(sentencesTrainIn)
            for i=1:size(sentencesTrain,1)
               sentencesTrainIn{i}=squeeze(sentencesTrain(i,:,:));
            end
            size(sentencesTrainIn)
            for i=1:size(sentencesTest,1)
               sentencesTestIn{i}=squeeze(sentencesTest(i,:,:));
            end
            inputSize = size(sentencesTest,2)*size(sentencesTest,3);
            
            %% Train autoencoder *2 , create deepnet, get classification results
            % do supervised learning
            autoenc1 = trainAutoencoder(sentencesTrainIn,hiddenSize1, ...
            'MaxEpochs',400, ...
            'L2WeightRegularization',0.004, ... %impact of L2 reglarizer on network weights
            'SparsityRegularization',4, ... %impact sparcity regularizer, constrains sparsity of hidden layer output
            'SparsityProportion',0.15, ...%each hidden layer neuron proportion that output
            'ScaleData', false); 
            view(autoenc1)
            features1=encode(autoenc1, sentencesTrainIn);
            autoenc2 = trainAutoencoder(features1,hiddenSize2, ...
                'MaxEpochs',100, ...
                'L2WeightRegularization',0.002, ...
                'SparsityRegularization',4, ...
                'SparsityProportion',0.1, ...
                'ScaleData', false);
            features2 = encode(autoenc2,features1);
            softnet = trainSoftmaxLayer(features2,sentencesIdsTrain,'MaxEpochs',400);
            deepnet = stack(autoenc1,autoenc2,softnet);
            
            % Turn the training images into vectors and put them in a matrix
            xTrain = zeros(inputSize,numel(sentencesTrainIn));
            for i = 1:numel(sentencesTrainIn)
                xTrain(:,i) = sentencesTrainIn{i}(:);
            end
            xTest = zeros(inputSize,numel(sentencesTestIn));
            for i = 1:numel(sentencesTestIn)
                xTest(:,i) = sentencesTestIn{i}(:);
            end
            %Create all input to get features at end of training
            xAll=zeros(inputSize,(numel(sentencesTestIn)+numel(sentencesTrainIn)));
            for i=1:numel(sentencesTrainIn)
                xAll(:,i)= xTrain(:,i);
            end
            for i=1+numel(sentencesTrainIn):(numel(sentencesTestIn)+numel(sentencesTrainIn))
                xAll(:,i)= xTest(:,i);
            end
            sentenceIds=[sentencesIdsTrain, sentenceIdsTest];
            
            
            % Get prelim results of classifications in confusion matrix
            fprintf('Confusion matrix before fine tuning');
            testLabelPredictions = deepnet(xTest);
            plotconfusion(sentencesIdTest,testLabelPredictions);
            
            % Perform fine tuning
            deepnet = train(deepnet,xTrain,sentencesIdTrain,'useParallel','yes','showResources','yes');

            %Confusion matrix after fine tuning
            fprintf('Confusion matrix after fine tuning');
            testLabelPredictions = deepnet(xTest);
            plotconfusion(sentencesIdTest,testLabelPredictions);
            
            deepnet.Layers
            featureLayer = 'fc2';
            fprintf('Now extracting all features from layer fc2');
            allFeatures = activations(deepnet,imagesAll,featureLayer);
            size(allFeatures)
            save('helloFeatures.mat','allFeatures');
            sentences4=allFeatures;
        end
            
        
    end
    
    switch options.featureExtractionMethod
        case 'AUTOENCODE1'
            sentences=sentences2;
        case 'AUTOENCODE2'
            sentences=sentences3;
        case 'AUTOENCODE3'       
            sentences=sentences4;
    fprint('THe size of the features extracted sentences are');
    size(sentences)
end