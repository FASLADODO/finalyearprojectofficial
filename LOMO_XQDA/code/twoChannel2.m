%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Implements a 2 channel regression machine learning, producing 1 or 0 for matches
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
    threshAim=1;
    
    %% Assert that gal and probe labels match (they should)
    'WHether the gal and prob, sentences and images are matching labels'
    isequal(galClassLabel,probClassLabel)

    %% sort gal and prob labels
    %galClassLabel(1:8)
    %[galClassLabel,idx]=sort(galClassLabel);
    %probClassLabel=probClassLabel(idx);
    %galFea=galFea(idx);
    %probFea=probFea(idx);
    %galClassLabel(1:8)
    
    'initial gal, prob sizes'
    size(galFea)
    numMatches=0;
    reps=int16(length(galClassLabel)/length(unique(galClassLabel)))*2; %should be 4
    fprintf('\nTHe number of repetitions of sentenceIds are %d\n',reps);
    %galFea1 = galFea(p(1:int16(numMatches*(reps-1)/reps)), : );
    %probFea1 = probFea(p(1:int16(numMatches*(reps-1)/reps)), : );
    %classLabelGal1=galClassLabel(p(1:int16(numMatches*(reps-1)/reps)));
    %classLabelProb1=probClassLabel(p(1:int16(numMatches*(reps-1)/reps)));
    
    
    if(~options.trainAll)
        numMatches=int16(size(galFea,1)*(reps-1)/reps);
    else
       numMatches=size(galFea,1); 
    end   
    numTestMatches=int16(size(galFea,1)*(reps-1)/reps);
    fprintf('generating galFea1, probFea1, with correct match labels, in proportion to falsePositiveRatio\n')
    
    
    
    %% Create training positive matches
    p = randperm(size(galFea,1));
    galFea1 = galFea(p(1:numMatches), : );
    probFea1 = probFea(p(1:numMatches), : );
    classLabelGal1=galClassLabel(p(1:numMatches));
    classLabelProb1=probClassLabel(p(1:numMatches));
    matchResults=ones(numMatches,1)*threshAim;
    temp=zeros((numMatches*options.falsePositiveRatio),1);
    matchResults((1+numMatches):((1+options.falsePositiveRatio)*numMatches),1)=temp;
    
    testSize=min(options.testSize, int16(size(galFea,1)/reps));
    %% Squeeze removes singleton dimensions
    %for i=1:testSize
      %  s=randperm(reps/2);
      %  idx=s*int16(size(galFea
    %end
    
    galFeaT = galFea(p(numTestMatches+1 :end), : );
    probFeaT = probFea(p(numTestMatches+1 : end), : );
    classLabelGalT=galClassLabel(p(numTestMatches+1 : end));
    classLabelProbT=probClassLabel(p(numTestMatches+1 :end));
   size(classLabelGalT)
  [idsNot, indexes,notused]=unique(squeeze(classLabelGalT));
    indexes
    classLabelGal2=classLabelGalT(indexes(1:testSize));
    classLabelProb2=classLabelProbT(indexes(1:testSize));
    galFea2=galFeaT(indexes(1:testSize),:);
    probFea2=probFeaT(indexes(1:testSize),:);
    %classLabelGal2=classLabelGalT(1:testSize);
    %classLabelProb2=classLabelProbT(1:testSize);
    %galFea2=galFeaT(1:testSize,:);
    %probFea2=probFeaT(1:testSize,:);    

    'sizes of classLabelGal2 unique'
    size(classLabelGal2)
    size(unique(classLabelGal2))   
    fprintf('Size of training data is %d, reps %d, unique ids %d and falsepositiveratio %d\n',numMatches*(1+options.falsePositiveRatio), reps, length(unique(galClassLabel)),options.falsePositiveRatio)
    fprintf('Size of testing data is %d',length(classLabelGal2));
    
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
    %size(galFea1) %should be no.examples, no.features
    %size(probFea1)
    half=(size(probFea,2)/2);
    fprintf('Assembling joint input for trainingNetwork2');
    for i=1:numMatches*(options.falsePositiveRatio+1)
       % size(galFea1(i,:))
       % size(probFea1(i,:))
        trainingPairs(:,1,1,i)=(galFea1(i,:)-mean2(galFea1(i,:)))/std2(galFea1(i,:));
       % size(trainingPairs)
        trainingPairs(:,2,1,i)=(probFea1(i,:)-mean2(probFea1(i,:)))/std2(probFea1(i,:));
       % size(trainingPairs)
    end
    t0 = tic;
    'size of trainingPairs is'
    size(trainingPairs)    
    
    %% Create and train net
    %setdemorandstream(391418381) Optional setting of random params 
    %{
    new=patternnet(10);
    view(net);
    [net,tr] = train(net,trainingPairs,matchResults);
    nntraintool
    plotperform(tr)
    %}
    
    %
    %% Create and train a better net
    noFeatsIn=size(galFea,2);
    
    layers = [ ...
    imageInputLayer([size(trainingPairs,1) 2 1 ], 'Name', 'input1');
    convolution2dLayer([4,2], 15 , 'Name', 'convol1'); %25 filters,  2 width height 1
    reluLayer('Name', 'relu1');

   % maxPooling2dLayer([2,1], 'Name', 'maxpool1'); %2 width height 1, moves 2 along horizontally, 0 vertically
   % convolution2dLayer([2,1], 15, 'Name', 'convol2'); %25 filters,  2 width height 1
   % reluLayer('Name', 'relu2');
   % convolution2dLayer([3,1],20,'Name','convol3');
    fullyConnectedLayer(5, 'Name', 'fulll1');
    reluLayer('Name', 'relu3');
    fullyConnectedLayer(2, 'Name', 'finalOutPut'); %fully connected layer of size 1
    softmaxLayer;
    classificationLayer();
%    regressionLayer()
]; % The software determines the size of the output during training.

    layers
    netOptions = trainingOptions('sgdm','InitialLearnRate',options.learningRate, ...
        'MaxEpochs',options.maxEpochs,'Verbose',true);%can onle select ecxecutionenvironment in 2017
    %lets move to autoencoder then see what have in computing labs
    %size(trainingPairs)
    %size(matchResults)
    fprintf('Training network twoChannel2\n');
    %trainingPairs(:,:,:,1)
    deepnet = trainNetwork(trainingPairs,categorical(matchResults),layers,netOptions);

    



    trainTime = toc(t0);

    
    
                    
                    t0=tic;
                    %% Test every combination and store in results dist matrix
                    for i = 1:size(galFea2,1)
                        %fprintf('Currently tested %d/%d\n', i, size(galFea2,1))
                        for u=1:size(probFea2,1)
                            temp1=(galFea2(i,:)-mean2(galFea2(i,:)))/std2(galFea2(i,:));
                            temp2=(probFea2(u,:)-mean2(probFea2(u,:)))/std2(probFea2(u,:));
                            testPairs(:,1,1,1)=temp1;
                            testPairs(:,2,1,1)=temp2;
                            
                            values = activations(deepnet,testPairs,'finalOutPut');
                            %values=predict(net,testPairs);
                            match=classLabelGal2(i)==classLabelProb2(u);
                            %2 numbers 1e-04 i think 1st is prob 0 second
                            %is prob 1 
                            dist(i,u)=abs(threshAim-values(2));  %ones down centre should match
                            
                            if(match && i~=1 && u~=1 && i<10)
                               fprintf('match %0.2f and nearest wrong neighbour %0.2f \n',dist(i,u), dist(i,u-1)) 
                            end
                        end
                    end   
                    matchTime = toc(t0); 
        %% Verbose feedback
    fprintf('Fold %d: ', iter);
    fprintf('Training time: %.3g seconds. ', trainTime);    
    fprintf('Matching time: %.3g seconds.\n', matchTime); 
                    
        %{            
    %% Create test pairs
    testPairs=zeros(size(galFea2,2),2,1,size(galFea2,1).^2);
    testMatches=zeros(size(galFea2,1).^2,1);
    for gal = 1 : size(galFea2,1)
       for prob= 1:size(galFea2,1)
          testPairs(:,1,1,(gal-1)*size(galFea2,1)+prob)=galFea2(gal,:).'; 
          testPairs(:,2,1,(gal-1)*size(galFea2,1)+prob)=probFea2(prob,:).'; 
          testMatches((gal-1)*size(galFea2,1)+prob)= isequal(classLabelGal2(gal), classLabelProb2(prob));
       end
    end
    %% Get results
    t0=tic;
    
    testResults=zeros(size(galFea2,1),size(galFea2,1));
    %results=net(testPairs);
    size(testPairs)
    results=predict(net,testPairs);
    %distance between correct and predicted
    %results=abs(testMatches-results); DONT WANT, CLOSER TO 1=BETTER
    %we do distance and it is a measure of 0-nomatch 1-match therefore
    %abs(1-result)
    results=abs(1-results);
    for i=1:length(results)
       results(i)=abs(1-results(i));
    end
    dist=reshape(results, size(testResults));
    matchTime = toc(t0);      

    %% Verbose feedback
    fprintf('Fold %d: ', iter);
    fprintf('Training time: %.3g seconds. ', trainTime);    
    fprintf('Matching time: %.3g seconds.\n', matchTime); 
    %}
    
end
