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
function [dist,classLabelGal2, classLabelProb2]=feedForwardMatch(galFea, probFea,galClassLabel,probClassLabel, iter, options)
    testSize=options.testSize;
    threshAim=1;
    
    %% Assert that gal and probe labels match (they should)
    'WHether the gal and prob, sentences and images are matching labels'
    isequal(galClassLabel,probClassLabel)
    
    fprintf('generating galFea1, probFea1, with correct match labels, in proportion to falsePositiveRatio')
    %% Create training positive matches
    numMatches=size(galFea,1);
    if(~options.trainAll)
        numExamples=int16(numMatches/2);
    else
       numExamples=numMatches; 
    end
    
    p = randperm(numMatches);
    galFea1 = galFea(p(1:numExamples), : );
    probFea1 = probFea(p(1:numExamples), : );
    classLabelGal1=galClassLabel(p(1:numExamples));
    classLabelProb1=probClassLabel(p(1:numExamples));
    matchResults=ones(numExamples,1)*threshAim;
    temp=zeros((numExamples*options.falsePositiveRatio),1);
    matchResults((1+numExamples):((1+options.falsePositiveRatio)*numExamples),1)=temp;
    
    
    %% Now add training false matches
    for i= 1: numExamples
        falseMatchIndexes=find(classLabelProb1 ~= classLabelGal1(i));
        n = randperm(length(falseMatchIndexes));
        for f=1:options.falsePositiveRatio
            galFea1(i+numExamples*f,:)=galFea1(i,:);
            probFea1(i+numExamples*f,:)= probFea1(falseMatchIndexes(n(f)),:);%Take random negative match  
        end
    end
    %can go through pairs by first index, creates two columns, each col
    %represents an image/sentence feature set
    %trainingPairs= cat(3,galFea1,probFea1);
    size(galFea1) %should be no.examples, no.features
    size(probFea1)
    half=(size(probFea,2)/2);
    fprintf('Assembling joint input for feedforwardmatch');
    for i=1:numExamples*(options.falsePositiveRatio+1)
        size(galFea1(i,:))
        size(probFea1(i,:))
        trainingPairs(:,1,i)=(galFea1(i,:)-mean2(galFea1(i,:)))/std2(galFea1(i,:));
        size(trainingPairs)
        trainingPairs(:,2,i)=(probFea1(i,:)-mean2(probFea1(i,:)))/std2(probFea1(i,:));
        size(trainingPairs)
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
    
    %
    %% Create and train a better net
    noFeatsIn=size(galFea,2);
    
  
    size(trainingPairs)
    size(matchResults)
    fprintf('Training network feedForward\n');
    inputSize=size(trainingPairs,2)*size(trainingPairs,1);
    xTrain = zeros(inputSize,size(trainingPairs,3));
    for i = 1:size(trainingPairs,3)
    	xTrain(:,i) = reshape(trainingPairs(:,:,i), inputSize,1);
    end
    
    
    net = feedforwardnet([40,20,10]);
    
    size(xTrain)
    [net, tr] = train(net, xTrain, matchResults.');
    view(net)
 
    trainTime = toc(t0);
    testSize=min(options.testSize, int16(numMatches/2));
    %% Squeeze removes singleton dimensions
    galFea2 = galFea(p(int16(numMatches/2)+1 :int16(numMatches/2)+ testSize), : );
    probFea2 = probFea(p(int16(numMatches/2)+1 : int16(numMatches/2)+testSize), : );
    classLabelGal2=galClassLabel(p(int16(numMatches/2)+1 : int16(numMatches/2)+testSize));
    classLabelProb2=probClassLabel(p(int16(numMatches/2)+1 :int16(numMatches/2)+ testSize));
    
    dist=zeros(size(galFea2,1),size(probFea2,1));
                    
                    t0=tic;
                    %% Test every combination and store in results dist matrix
                    for i = 1:size(galFea2,1)
                        %fprintf('Currently tested %d/%d\n', i, size(galFea2,1))
                        for u=1:size(probFea2,1)
                            temp1=(galFea2(i,:)-mean2(galFea2(i,:)))/std2(galFea2(i,:));
                            temp2=(probFea2(u,:)-mean2(probFea2(u,:)))/std2(probFea2(u,:));
                            %testPairs(:,1)=temp1;
                            %testPairs(:,2)=temp2;
                            testPairs=[temp1,temp2].';
                          
                            values= net(testPairs);
                            match=classLabelGal2(i)==classLabelProb2(u);
                            dist(i,u)=abs(threshAim-values);  %ones down centre should match
                            
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