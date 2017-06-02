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
function [dist,classLabelGal2, classLabelProb2]=twoChannel(galFea, probFea,galClassLabel,probClassLabel, iter, options)
    
    
    %% Assert that gal and probe labels match (they should)
    'WHether the gal and prob, sentences and images are matching labels'
    isequal(galClassLabel,probClassLabel)
    
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
    for i=1:int16(numMatches/2)*(options.falsePositiveRatio+1)
        trainingPairs(:,1,1,i)=galFea1(i,:);
        %trainingPairs(:,2,1,i)=galFea1(i,:);
        size(trainingPairs(:,1,i));
        size(galFea1(i,:));
        size(probFea1(i,:));
        %trainingPairs(:,1,2,i)=probFea1(i,:);
        trainingPairs(:,2,1,i)=probFea1(i,:);
    end
    t0 = tic;
        
    %% Create and train net
    %setdemorandstream(391418381) Optional setting of random params 
    %{
    new=patternnet(10);
    view(net);
    [net,tr] = train(net,trainingPairs,matchResults);
    nntraintool
    plotperform(tr)
    %}
    
    %% Create and train a better net
    layers = [ ...
    imageInputLayer([size(galFea,2) 2 1]);
    convolution2dLayer([2,1],25); %25 filters,  2 width height 1
    reluLayer();
    maxPooling2dLayer([2,1],'Stride',2); %2 width height 1, moves 2 along horizontally, 0 vertically
    fullyConnectedLayer(1) %fully connected layer of size 1
    regressionLayer];

    options = trainingOptions('sgdm','InitialLearnRate',0.001, ...
        'MaxEpochs',15);
    size(trainingPairs)
    size(matchResults)
    net = trainNetwork(trainingPairs,matchResults,layers,options)
    




    trainTime = toc(t0);
    
    %% Squeeze removes singleton dimensions
    galFea2 = galFea(p(int16(numMatches/2)+1 : end), : );
    probFea2 = probFea(p(int16(numMatches/2)+1 : end), : );
    classLabelGal2=galClassLabel(p(int16(numMatches/2)+1 : end));
    classLabelProb2=probClassLabel(p(int16(numMatches/2)+1 : end));
    
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
    results=predict(net,testPairs);
    %distance between correct and predicted
    results=abs(testMatches-results);
    dist=reshape(results, size(testResults));
    matchTime = toc(t0);      

    %% Verbose feedback
    fprintf('Fold %d: ', iter);
    fprintf('Training time: %.3g seconds. ', trainTime);    
    fprintf('Matching time: %.3g seconds.\n', matchTime); 
    
end