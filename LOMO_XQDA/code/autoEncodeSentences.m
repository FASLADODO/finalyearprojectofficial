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
%Added mat2cell, as trying to put inputs into cell array each cell contains
%matrix

%TO DO:
%ORGANISE FOR PLOTCONFUSION, FIRST 10 CLASSES HELD IN FIRST 10 EXAMPLES
%MIGHT WANT TO SEE IF CLASSIFICATION DONE BY MAXIMUM OR BY DISTANCE.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [sentences,sentenceIds]=autoEncodeSentences(sentences, sentenceIds, options)
    'sentenceIds are class'
    class(sentenceIds)
    %% Encoder parameters
    rng('default')%explicit set random seed, so results replicable
    hiddenSize1 = size(sentences,4)%size of hidden layer in autoencoder, want smaller than sentences
    hiddenSize2=50%int16(0.5*size(sentences,4))
    
    
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
            %sentencesIn{1}
            %sentencesIn{2} dont know why mat2cell doesnt work, maybe as it
            %makes everything into a cell?? mat2cell(squeeze(sentences(config,i,:,:)),52,200);
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
            size(sentenceIdsProcess)
            sentencesIdsTrain=sentenceIdsProcess(indexes);
            
            %% NOW CONVERTING SENTENCE IDS TO HOTCODING
            sentencesIdsHot=hotCoding(sentenceIdsProcess);
            sentencesIdsTrain2=sentencesIdsHot(indexes,:);
            
            %for dd= 1:length(indexes)
               % temp=int2str(sentenceIdsProcess(indexes(dd)));
              %  temp
               % sentencesIdsTrain(dd,1)=temp;
            %end
            %sentencesIdsTrain=num2str(sentenceIdsProcess(indexes));
            testIndexes= setdiff([1:size(sentencesProcess,1)],indexes);
            sentencesTest=sentencesProcess(testIndexes,:,:);
            %for dd= 1:length(testIndexes)
             %   sentencesIdsTrain(dd)=int2str(sentenceIdsProcess(testIndexes(dd)));
            %end
            sentencesIdsTest= sentenceIdsProcess(testIndexes);
            sentencesIdsTest2=sentencesIdsHot(testIndexes,:);
            
            
            
            %sentencesIdsTest=num2str(sentenceIdsProcess(testIndexes));
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
            inputSize = size(sentencesTest,2)*size(sentencesTest,3);%number words, number vecs per word
            
            %% Train autoencoder *2 , create deepnet, get classification results
            % do supervised learning
            autoenc1 = trainAutoencoder(sentencesTrainIn,hiddenSize1, ...
            'MaxEpochs',10, ...%200
            'L2WeightRegularization',0.004, ... %impact of L2 reglarizer on network weights
            'SparsityRegularization',4, ... %impact sparcity regularizer, constrains sparsity of hidden layer output
            'SparsityProportion',0.15, ...%each hidden layer neuron proportion that output
            'ScaleData', false); 
            view(autoenc1)
            features1=encode(autoenc1, sentencesTrainIn);
            autoenc2 = trainAutoencoder(features1,hiddenSize2, ...
                'MaxEpochs',5, ...%100
                'L2WeightRegularization',0.002, ...
                'SparsityRegularization',4, ...
                'SparsityProportion',0.1, ...
                'ScaleData', false);
            features2 = encode(autoenc2,features1);
            'sizes features2 and sentencesIdsTrain2 conjugate'
            size(features2)
            size(sentencesIdsTrain2.')
            %both meant to be kby n mby n data
            %needs to be double/single format for ids
            softnet = trainSoftmaxLayer(features2,sentencesIdsTrain2.','MaxEpochs',100);
            deepnet = stack(autoenc1,autoenc2,softnet);
            
            % Turn the training sentences into vectors and put them in a matrix
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
            fprintf('size train, test, all data')
            'x train data size'
            size(xTrain)
            'x test data size'
            size(xTest)
            'x all datasize'
            size(xAll)
            
            for i=(1+numel(sentencesTrainIn)):(numel(sentencesTestIn)+numel(sentencesTrainIn))
                xAll(:,i)= xTest(:,i-numel(sentencesTrainIn));
            end
            sentenceIds=[sentencesIdsTrain; sentencesIdsTest];
            
            
            % Get prelim results of classifications in confusion matrix
            fprintf('Confusion matrix before fine tuning');
            testLabelPredictions = deepnet(xTest);
            'size sentenceidstest2'
            size(sentencesIdsTest2)%317 1360
            'size testlabelpredictions'
            size(testLabelPredictions)%1360 by 317
            %sentencesIdsTest2(1,:).' %Orig 317 by 1360, each row represents an example. 1360 classes, 317 examples
            origLabels=sentencesIdsTest2.';
            %testLabelPredictions(:,1)
            
            %reorganise using IDX SO THAT FIRST 10 CLASSES HELD IN FIRST 10
            %EXAMPLES.
            
            plotconfusion(origLabels(1:10,1:10),testLabelPredictions(1:10,1:10));
            %N by M, number of classes, number of examples
            size(xTrain)
            %size(sentencesIdsTrain)
            % Perform fine tuning
            deepnet = train(deepnet,xTrain,sentencesIdsTrain2.','useParallel','yes','showResources','yes');
            %autoenc1.DecoderWeights=
            %autoenc1.DecoderBiases=
            %autoenc1.encoderWeights
            %autoenc1.encoderBiases
            %Confusion matrix after fine tuning
            fprintf('Confusion matrix after fine tuning');
            %testLabelPredictions = deepnet(xTest);
            %plotconfusion(sentencesIdsTest2(1:10,1:50).',testLabelPredictions(1:50,1:10));
            
            
            %deepnet
            %deepnet.Layers
            %deepnet.Outputs
            %deepnet.Outputs{1,1}
            %deepnet.Outputs{1,2}
            %deepnet.Outputs{1,3}
            deepnet.outputConnect=[0, 1, 0];
            features2=deepnet(xAll);
            %featureLayer = 'fc2';
            fprintf('Now extracting all features from layer first');
            %layer1=deepnet.Layers{1}
            %layer2=deepnet.Layers{2}
            %resy=activations(deepnet, xAll, 2)
            %features1=layer1(xAll);
            %features1 = encode(deepnet{1,1},xAll);non cell array obj apaz
            %5size(features1)
            %features2=layer2(features1);
            %features2 = encode(deepnet{2,1},features1);
            %size(features2)
            %allFeatures = activations(deepnet,xAll,featureLayer);
            %size(allFeatures)
            save('helloFeatures.mat','features2');
            sentences4(config,:,:)=features2.';
        end
            
        
    end
    
    switch options.featureExtractionMethod
        case 'AUTOENCODE1'
            sentences=sentences2;
        case 'AUTOENCODE2'
            sentences=sentences3;
        case 'AUTOENCODE3'       
            sentences=sentences4;
    fprintf('THe size of the features extracted sentences are');
    size(sentences)
end