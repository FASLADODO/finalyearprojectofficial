function [personIds,images]=autoEncodeImages(images, personIds, options)

    %images(:,:,:,i)
    %% Encoder parameters
    rng('default');%explicit set random seed, so results replicable
    hiddenSize1 = options.hiddensize1;%size of hidden layer in autoencoder, want smaller than sentences
    hiddenSize2= options.hiddensize2;%int16(0.5*size(sentences,4))
        
        for i= 1:size(images,4)
           newImages(:,:,i)=squeeze(mean(rgb2gray(images(:,:,:,i)),3)); 
        end
        images=newImages;
    
        
        %%  If autoencoder is type 1 or 2 will just unsupervised extract features
        %if(strcmp(options.featureExtractionMethod, 'AUTOENCODE2') | strcmp(options.featureExtractionMethod, 'AUTOENCODE1'))
        
        if(options.trainLevel==1 || options.trainLevel==2)
            imagesIn=cell(size(images,3),1);
            for i=1:size(images,3)
               imagesIn{i}=squeeze(images(:,:,i));
            end
            size(imagesIn);
            %sentencesIn{1}
            %sentencesIn{2} dont know why mat2cell doesnt work, maybe as it
            %makes everything into a cell?? mat2cell(squeeze(sentences(config,i,:,:)),52,200);
            
                fprintf(' Running autoencoder1...\n');
                autoenc1 = trainAutoencoder(imagesIn,hiddenSize1, ...
                'MaxEpochs',options.maxepoch1, ...%400
                'L2WeightRegularization',0.004, ... %impact of L2 reglarizer on network weights
                'SparsityRegularization',4, ... %impact sparcity regularizer, constrains sparsity of hidden layer output
                'SparsityProportion',0.1, ...%each hidden layer neuron proportion that output
                'ScaleData', false); 
                view(autoenc1)
                feat1=encode(autoenc1, imagesIn);
                fprintf('the size of the features generated is %d %d \n', size(feat1,1), size(feat1,2));
                size(feat1);
                if(options.trainLevel==1)
                    images2=feat1.';
                end

            %Take these features to next layer, then reduce down , then
            %backtrain to make features with labelled data more accurate
            %% If AUTOENCODE2 reduce size of feature data further for more intense representation
            if(options.trainLevel==2)
               
                    fprintf('Running autoencoder2...\n');
                    autoenc2 = trainAutoencoder(feat1,hiddenSize2, ...
                        'MaxEpochs',options.maxepoch2, ...%100
                        'L2WeightRegularization',0.002, ...
                        'SparsityRegularization',4, ...
                        'SparsityProportion',0.1, ...
                        'ScaleData', false);
                    feat2 = encode(autoenc2,feat1);
                    size(feat2)
                        images3=feat2.';

            end
        end
        %% If autoencode3 will be extra training deep neural network, will need to split data to train and test
        if(options.trainLevel==3)
                            
                %% Order sentences
                [imagesIdsProcess,idx]=sort(personIds);
                imagesProcess=squeeze(images(:,:,idx));%all the files, sentences,words, word vectors
                fprintf('\nthe size of imagesprocess is (%d, %d)\n', size(imagesProcess,1),size(imagesProcess,2));
                
                occur=0;
                indexes=[];
                idx=1;
                old=0;
                %%Construct sentencesTest and sentencesTrain based on settings,
                % Get indexes for train data
                switch options.imageSplit
                    case 'pairs'
                            fprintf('Training data is all pairs that exist of imageData\n');
                            for i=1:length(imagesIdsProcess)
                                if(imagesIdsProcess(i)~=old && occur>1)
                                    for p=1:2%occur
                                       indexes(idx)=i-p;
                                       idx=idx+1;                               
                                    end      
                                    occur=1;
                                    old=imagesIdsProcess(i);
                                else
                                    if(imagesIdsProcess(i)==old)
                                        occur=occur+1;
                                    else
                                       old=imagesIdsProcess(i);
                                       occur=1;
                                    end
                                end
                            end   
                    case 'oneofeach'
                            fprintf('Training data is one of each of all image data');
                           for i=1:length(imagesIdsProcess)
                                if(imagesIdsProcess(i)~=old && occur>1)

                                    indexes(idx)=i-p;
                                    idx=idx+1;

                                    occur=1;
                                    old=imagesIdsProcess(i);
                                else
                                    if(imagesIdsProcess(i)==old)
                                        occur=occur+1;
                                    else
                                       old=imagesIdsProcess(i);
                                       occur=1;
                                    end
                                end
                            end

                    case 'oneofeach+'
                           fprintf('Training data is one of each of all image data + extras');
                           for i=1:length(imagesIdsProcess)
                                if(imagesIdsProcess(i)~=old && occur>1)

                                    indexes(idx)=i-p;
                                    idx=idx+1;

                                    occur=1;
                                    old=imagesIdsProcess(i);
                                else
                                    if(imagesIdsProcess(i)==old)
                                        occur=occur+1;
                                    else
                                       old=imagesIdsProcess(i);
                                       occur=1;
                                    end
                                end
                            end                   
                            indexes= setdiff([1:size(imagesProcess,3)],indexes);
                end       
                %create sentencesTrain and sentencesTest
                %sentneceProcess is all current data in this configfile
                imagesTrain=imagesProcess(:,:,indexes);
                %sentencesIdsTrain=sentenceIdsProcess(indexes);

                %% NOW CONVERTING SENTENCE IDS TO HOTCODING
                imagesIdsHot=hotCoding(imagesIdsProcess);
                imagesIdsTrain=imagesIdsHot(indexes,:)
                testIndexes= setdiff([1:size(imagesProcess,3)],indexes);
                imagesTest=imagesProcess(:,:,testIndexes);
                %sentencesIdsTest= sentenceIdsProcess(testIndexes);
                imagesIdsTest=imagesIdsHot(testIndexes,:);
                %size(sentencesIdsTest)
                %size(sentencesTest)
                %% adapt to size sentenceidtest set two is for new created test adn train
                imagesIdsTrain2=imagesIdsTrain(1:min(options.imageTrainSplit,size(imagesIdsTrain,1)),:);
                imagesTrain2=imagesTrain(:,:,1:min(options.imageTrainSplit,size(imagesTrain,3)));
                positions=ismember(imagesIdsTest,imagesIdsTrain2, 'rows');
                
                imagesTest2=imagesTest(:,:,positions);
                imagesIdsTest2=imagesIdsTest(positions,:);
                
                
                %sentencesIdsTest=num2str(sentenceIdsProcess(testIndexes));
                fprintf('The size of imagesTest is (%d %d)\n', size(imagesTest,1), size(imagesTest,2))
                fprintf('The size of imagesTrain is (%d %d)\n', size(imagesTrain,1), size(imagesTrain,2))           
               

                imagesTrainIn=cell(size(imagesTrain2,3),1);
                imagesTestIn=cell(size(imagesTest2,3),1);
                imagesAllIn=cell(size(imagesProcess,3),1);
                fprintf('The size of imagesTrainIn is (%d %d)\n', size(imagesTrainIn,1), size(imagesTrainIn,2))  
               
                for i=1:size(imagesProcess,3)
                   imagesAllIn{i}=squeeze(imagesProcess(:,:,i));
                end
                %trainingIn and testingIn based off subset
                %options.sentenceTrainSplit
                for i=1:size(imagesTrain2,3)
                   imagesTrainIn{i}=squeeze(imagesTrain2(:,:,i));
                end
                
                for i=1:size(imagesTest2,3)
                   imagesTestIn{i}=squeeze(imagesTest2(:,:,i));
                end
                inputSize = size(imagesTest2,1)*size(imagesTest2,2);%number words, number vecs per word

                %% Train autoencoder *2 , create deepnet, get classification results
                % do supervised learning
                fprintf('Running autoencoder1...\n');
                autoenc1 = trainAutoencoder(imagesTrainIn,hiddenSize1, ...
                'MaxEpochs',options.maxepoch1, ...%200
                'L2WeightRegularization',0.004, ... %impact of L2 reglarizer on network weights
                'SparsityRegularization',4, ... %impact sparcity regularizer, constrains sparsity of hidden layer output
                'SparsityProportion',0.15, ...%each hidden layer neuron proportion that output
                'ScaleData', false); 
                view(autoenc1)
                fprintf('Running autoencoder2...\n');
                features1=encode(autoenc1, imagesTrainIn);
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
                size(features2)
                size(imagesIdsTrain2)
                size(imagesTrainIn)
                
                softnet = trainSoftmaxLayer(features2,imagesIdsTrain2.','MaxEpochs',options.maxepoch3);
                deepnet = stack(autoenc1,autoenc2,softnet);

                size(imagesTrainIn{1})
                
                % Turn the training sentences into vectors and put them in a matrix
                %xTrain = zeros(inputSize,numel(imagesTrainIn));
                xTrain = zeros(inputSize,numel(imagesTrainIn));
                for i = 1:numel(imagesTrainIn)
                    xTrain(:,i) = imagesTrainIn{i}(:);
                end
                %xTrain=imagesTrain2;
                xTest = zeros(inputSize,numel(imagesTestIn));
                for i = 1:numel(imagesTestIn)
                    xTest(:,i) = imagesTestIn{i}(:);
                end
                %xTest=imagesTest2;
                %Create all input to get features at end of training
                xAll=zeros(inputSize,numel(imagesAllIn));
                %%%xAll=zeros(inputSize,(numel(sentencesTestIn)+numel(sentencesTrainIn)));
                for i=1:numel(imagesAllIn)
                    xAll(:,i)= imagesAllIn{i}(:);
                end
                %xAll=imagesProcess;

                %sentenceIds=[sentencesIdsTrain; sentencesIdsTest];
                personIds=imagesIdsProcess;

                %If pre-trained network does not exist, train it
                %if (exist(netSentences{config}, 'file') ~= 2 || options.force)
                % Get prelim results of classifications in confusion matrix
                    fprintf('Confusion matrix before fine tuning');
                    if(size(xTest,2)~=0)
                        testLabelPredictions = deepnet(xTest);
                        origLabels=imagesIdsTest2.';           

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
                    size(xTrain)
                    size(squeeze(imagesTrainIn.'))
                    size(imagesIdsTrain2.')
                    deepnet.numInputs
                    deepnet = train(deepnet,xTrain,imagesIdsTrain2.');
                    if(size(xTest,2)~=0)
                        %Confusion matrix after fine tuning
                        fprintf('Confusion matrix after fine tuning');
                        testLabelPredictions = deepnet(xTest);
                        figure;
                        imagesIdsTest2(1:noPlots,resultIndexes).';
                        plotconfusion(imagesIdsTest2(1:noPlots,resultIndexes).',testLabelPredictions(resultIndexes,1:noPlots));
                    end
                    deepnet.outputConnect=[0, 1, 0];
                    %fprintf('Net %s already exists, loading...\n',netSentences{config});
                    %save(netSentences{config}, 'deepnet');
                %else
                    
                  % load( netSentences{config});
                %end
                
                features2=deepnet(xAll);
                size(xAll)
                size(features2)
                %featureLayer = 'fc2';
                fprintf('Now extracting all features from layer first\n');
                %save(saveSentences{config}, 'features2');
            %else
             %   fprintf('Sentences %s already exists, loading...\n',saveSentences{config});
             %   load(saveSentences{config});
           % end
            images4=features2.';
        end
            
        
   
    
    switch options.trainLevel
        case 1
            images=images2;
        case 2
            images=images3;
        case 3       
            images=images4;
    fprintf('THe size of the features extracted images are');
    size(images)
end