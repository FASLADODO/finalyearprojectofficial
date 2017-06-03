%THIS IS BASIC BY PRODUCING
%Sentences are 4D sentenceConfig, sentence, word, wordvector
%Want to produce 3D sentenceconfig, sentence, sentencevector
%% NEED TO SAVE NETS, NEED TO ADD ALTERING SENTENCETRAINSIZE
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
%NEED TO FIX TRAINING SO TRAINING DATA AND TEST DATA CREATED, AND CULUM
%KEPT TO PRODUCE OUTPUT
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [sentences,sentenceIds]=autoEncodeSentences(sentences, sentenceIds, netSentences, saveSentences, options)
   % 'sentenceIds are class'
    %class(sentenceIds)
    %% Encoder parameters
    rng('default');%explicit set random seed, so results replicable
    hiddenSize1 = options.hiddensize1;%size of hidden layer in autoencoder, want smaller than sentences
    hiddenSize2= options.hiddensize2;%int16(0.5*size(sentences,4))
    
    
    for config = 1:size(sentences,1)
        
        %%  If autoencoder is type 1 or 2 will just unsupervised extract features
        %if(strcmp(options.featureExtractionMethod, 'AUTOENCODE2') | strcmp(options.featureExtractionMethod, 'AUTOENCODE1'))
        if(options.trainLevel==1 || options.trainLevel==2)
            sentencesIn=cell(size(sentences,2),1);
            for i=1:size(sentences,2)
               sentencesIn{i}=squeeze(sentences(config,i,:,:));
            end
            size(sentencesIn);
            %sentencesIn{1}
            %sentencesIn{2} dont know why mat2cell doesnt work, maybe as it
            %makes everything into a cell?? mat2cell(squeeze(sentences(config,i,:,:)),52,200);
            if (exist(saveSentences{config}, 'file') ~= 2 || options.force)
                fprintf('Sentences %s doesnt exist already, \n Running autoencoder1...\n',saveSentences{config});
                autoenc1 = trainAutoencoder(sentencesIn,hiddenSize1, ...
                'MaxEpochs',options.maxepoch1, ...%400
                'L2WeightRegularization',0.004, ... %impact of L2 reglarizer on network weights
                'SparsityRegularization',4, ... %impact sparcity regularizer, constrains sparsity of hidden layer output
                'SparsityProportion',0.1, ...%each hidden layer neuron proportion that output
                'ScaleData', false); 
                view(autoenc1)
                feat1=encode(autoenc1, sentencesIn);
                fprintf('the size of the features generated is %d %d \n', size(feat1,1), size(feat1,2));
                size(feat1);
                if(options.trainLevel==1)
                    save(saveSentences{config}, 'feat1');
                    sentences2(config,:,:)=feat1.';
                end
                
            else
                
                if(options.trainLevel==1)
                    fprintf('Sentences %s already exists, \n Loading... \n',saveSentences{config});
                    load(saveSentences{config},'feat1');
                    sentences2(config,:,:)=feat1.';
                end
            end
            

            %Take these features to next layer, then reduce down , then
            %backtrain to make features with labelled data more accurate
            %% If AUTOENCODE2 reduce size of feature data further for more intense representation
            if(options.trainLevel==2)
                if (exist(saveSentences{config}, 'file') ~= 2 || options.force)
                    fprintf('Sentences %s doesnt exist already, \n Running autoencoder2...\n',saveSentences{config});
                    autoenc2 = trainAutoencoder(feat1,hiddenSize2, ...
                        'MaxEpochs',options.maxepoch2, ...%100
                        'L2WeightRegularization',0.002, ...
                        'SparsityRegularization',4, ...
                        'SparsityProportion',0.1, ...
                        'ScaleData', false);
                    feat2 = encode(autoenc2,feat1);
                    size(feat2)
                    save(saveSentences{config}, 'feat2');
                     size(feat2)
                        sentences3(config,:,:)=feat2.';
                else
                    fprintf('Sentences %s already exists \n loading...\n',saveSentences{config});
                    load(saveSentences{config}, 'feat2');
                    sentences3(config,:,:)=feat2.';
                end
               
            end
        end
        %% If autoencode3 will be extra training deep neural network, will need to split data to train and test
        if(options.trainLevel==3)
            
            if (exist(saveSentences{config}, 'file') ~= 2 || options.force)
                
                %% Order sentences
                [sentenceIdsProcess,idx]=sort(sentenceIds);
                sentencesProcess=squeeze(sentences(config,idx,:,:));%all the files, sentences,words, word vectors
                fprintf('\nthe size of sentencesprocess is (%d, %d)\n', size(sentencesProcess,1),size(sentencesProcess,2));
                
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
                %sentencesIdsTrain=sentenceIdsProcess(indexes);

                %% NOW CONVERTING SENTENCE IDS TO HOTCODING
                sentencesIdsHot=hotCoding(sentenceIdsProcess);
                sentencesIdsTrain=sentencesIdsHot(indexes,:);
                testIndexes= setdiff([1:size(sentencesProcess,1)],indexes);
                sentencesTest=sentencesProcess(testIndexes,:,:);
                %sentencesIdsTest= sentenceIdsProcess(testIndexes);
                sentencesIdsTest=sentencesIdsHot(testIndexes,:);
                %size(sentencesIdsTest)
                %size(sentencesTest)
                %% adapt to size sentenceidtest set two is for new created test adn train
                sentencesIdsTrain2=sentencesIdsTrain(1:min(options.sentenceTrainSplit,size(sentencesIdsTrain,1)),:);
                sentencesTrain2=sentencesTrain(1:min(options.sentenceTrainSplit,size(sentencesTrain,1)),:,:);
                positions=ismember(sentencesIdsTest,sentencesIdsTrain2, 'rows');
                
                sentencesTest2=sentencesTest(positions,:,:);
                sentencesIdsTest2=sentencesIdsTest(positions,:);
                
                
                %sentencesIdsTest=num2str(sentenceIdsProcess(testIndexes));
                fprintf('The size of sentencesTest is (%d %d)\n', size(sentencesTest,1), size(sentencesTest,2))
                fprintf('The size of sentencesTrain is (%d %d)\n', size(sentencesTrain,1), size(sentencesTrain,2))           
               

                sentencesTrainIn=cell(size(sentencesTrain2,1),1);
                sentencesTestIn=cell(size(sentencesTest2,1),1);
                sentencesAllIn=cell(size(sentencesProcess,1),1);
                fprintf('The size of sentencesTrainIn is (%d %d)\n', size(sentencesTrainIn,1), size(sentencesTrainIn,2))  
               
                for i=1:size(sentencesProcess,1)
                   sentencesAllIn{i}=squeeze(sentencesProcess(i,:,:));
                end
                %trainingIn and testingIn based off subset
                %options.sentenceTrainSplit
                for i=1:size(sentencesTrain2,1)
                   sentencesTrainIn{i}=squeeze(sentencesTrain2(i,:,:));
                end
                
                for i=1:size(sentencesTest2,1)
                   sentencesTestIn{i}=squeeze(sentencesTest2(i,:,:));
                end
                inputSize = size(sentencesTest2,2)*size(sentencesTest2,3);%number words, number vecs per word

                %% Train autoencoder *2 , create deepnet, get classification results
                % do supervised learning
                fprintf('Sentences %s doesnt exist already, \n Running autoencoder1...\n',saveSentences{config});
                autoenc1 = trainAutoencoder(sentencesTrainIn,hiddenSize1, ...
                'MaxEpochs',options.maxepoch1, ...%200
                'L2WeightRegularization',0.004, ... %impact of L2 reglarizer on network weights
                'SparsityRegularization',4, ... %impact sparcity regularizer, constrains sparsity of hidden layer output
                'SparsityProportion',0.15, ...%each hidden layer neuron proportion that output
                'ScaleData', false); 
                view(autoenc1)
                fprintf('Running autoencoder2...\n');
                features1=encode(autoenc1, sentencesTrainIn);
                autoenc2 = trainAutoencoder(features1,hiddenSize2, ...
                    'MaxEpochs',options.maxepoch2, ...%100
                    'L2WeightRegularization',0.002, ...
                    'SparsityRegularization',4, ...
                    'SparsityProportion',0.1, ...
                    'ScaleData', false);
                features2 = encode(autoenc2,features1);
                %'sizes features2 and sentencesIdsTrain2 conjugate'
               % size(features2)
               % size(sentencesIdsTrain2.')
                %both meant to be kby n mby n data
                %needs to be double/single format for ids
                softnet = trainSoftmaxLayer(features2,sentencesIdsTrain2.','MaxEpochs',options.maxepoch3);
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
                xAll=zeros(inputSize,numel(sentencesAllIn));
                %%%xAll=zeros(inputSize,(numel(sentencesTestIn)+numel(sentencesTrainIn)));
                for i=1:numel(sentencesAllIn)
                    xAll(:,i)= sentencesAllIn{i}(:);
                end


                %sentenceIds=[sentencesIdsTrain; sentencesIdsTest];
                sentenceIds=sentenceIdsProcess;

                %If pre-trained network does not exist, train it
                if (exist(netSentences{config}, 'file') ~= 2 || options.force)
                % Get prelim results of classifications in confusion matrix
                    fprintf('Confusion matrix before fine tuning');
                    if(size(xTest,2)~=0)
                        testLabelPredictions = deepnet(xTest);
                        origLabels=sentencesIdsTest2.';           

                        %% reorganise using IDX SO THAT FIRST 10 CLASSES HELD IN FIRST 10
                        %EXAMPLES. i think already ordered
                        figure; 
                        %if the setting is pairs then test will be quite far down to
                        %start,
                        noPlots=min(20, size(origLabels,2));
                        resultIndexes=zeros(noPlots,1);
                        for i = 1:noPlots%size(origLabels,2)
                            resultIndexes(i)= find(origLabels(:,i));
                        end

                        plotconfusion(origLabels(resultIndexes,1:noPlots),testLabelPredictions(resultIndexes,1:noPlots));
                        %N by M, number of classes, number of examples
                    else
                        fprintf('Due to the size of training data chosen, there are no repeats to use for intermitent confusion plot testing');
                    end
                    fprintf(' Training autoencoders with examples...\n');
                    % Perform fine tuning
                    deepnet = train(deepnet,xTrain,sentencesIdsTrain2.','useParallel','yes','showResources','yes');
                    if(size(xTest,2)~=0)
                        %Confusion matrix after fine tuning
                        fprintf('Confusion matrix after fine tuning');
                        testLabelPredictions = deepnet(xTest);
                        figure;
                        sentencesIdsTest2(1:noPlots,resultIndexes).';
                        plotconfusion(sentencesIdsTest2(1:noPlots,resultIndexes).',testLabelPredictions(resultIndexes,1:noPlots));
                    end
                    deepnet.outputConnect=[0, 1, 0];
                    fprintf('Net %s already exists, loading...\n',netSentences{config});
                    save(netSentences{config}, 'deepnet');
                else
                    
                   load( netSentences{config});
                end
                features2=deepnet(xAll);
                %featureLayer = 'fc2';
                fprintf('Now extracting all features from layer first\n');
                save(saveSentences{config}, 'features2');
            else
                fprintf('Sentences %s already exists, loading...\n',saveSentences{config});
                load(saveSentences{config});
            end
            sentences4(config,:,:)=features2.';
        end
            
        
    end
    
    switch options.trainLevel
        case 1
            sentences=sentences2;
        case 2
            sentences=sentences3;
        case 3       
            sentences=sentences4;
    fprintf('THe size of the features extracted sentences are');
    size(sentences)
end