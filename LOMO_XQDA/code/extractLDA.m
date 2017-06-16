%Input rows are the sentences, cols are the variabls
function sentenceVectors=extractLDA(sentences, sentenceIds, numDims)

    idxWow=randperm(size(sentences,2));
    newSize=min(10000,size(sentences,2));%Newsize necessary to reduce memory usage
    sentences=sentences(:,idxWow(1:newSize));
    
    uniqSentenceIds=unique(sentenceIds);
    size(sentences)
    size(uniqSentenceIds)
    size(sentenceIds)
    sw=zeros(size(sentences,2),size(sentences,2));
    for u=1:length(uniqSentenceIds)
        withinClassIdx=find(sentenceIds==uniqSentenceIds(u));
        sentencesIn=sentences(withinClassIdx,:);
        meanC=mean(sentencesIn,1);
        %for all dimensions within this class
        for d=1:size(sentencesIn,1)
            sw=sw+(sentencesIn(d,:)-meanC).'*(sentencesIn(d,:)-meanC);  
        end
    end
    'size sw'
    size(sw)
    sb=zeros(size(sentences,2),size(sentences,2));
    meanOverall=mean(sentences,1);
    meanClass=zeros(length(uniqSentenceIds),size(sentences,2));
    for u= 1:length(uniqSentenceIds)
        withinClassIdx=find(sentenceIds==uniqSentenceIds(u));
        meanClass(u,:)=mean(sentences(withinClassIdx,:),1);        
    end

    for u=1:length(uniqSentenceIds)
       	 %sentencesIn=sentences(find(sentenceIds==uniqSentenceIds(u)),:);
           sb=sb+(meanClass(u,:)-meanOverall).'*(meanClass(u,:)-meanOverall);
    end
    %'size sb, then inv(sw)'
    %size(sb)
    %size(inv(sw))
    %orig=sw\sb;
    orig=pinv(sw)*sb
    %'sb first row'
    %sb(1:100,1)
    %'sw first row'
    %sw(1:100,1)
    %'inv sw first row'
    %temp=pinv(sw);
    %temp(1:100,1)
    %'orig'
    %orig(1:100,1)
    %meanS=mean(orig,1);%row vector
    %sentenceVector=bsxfun(@minus,orig,meanS);
    %covy=sentenceVector(idxWow(1:newSize),:).'*sentenceVector(idx(1:newSize),:)/size(sentenceVector,2);
    
    [eigVecs, eigVals]=eig(orig );%eigVecs cols are eigenVectors
    eigVals=diag(eigVals);%col vector
    eigVals=eigVals.';
    [eigVals,idx]=sort(eigVals,'descend');
    eigVecs(:,:)=eigVecs(:,idx);
    eigVecs=eigVecs(:,1:numDims);
    
    for s= 1: size(sentences,1)
        for i=1: numDims
            temp= sentences(s,:)*eigVecs(:,i);
            sentenceVectors(s,i)= temp;
        end
    end
    
end
