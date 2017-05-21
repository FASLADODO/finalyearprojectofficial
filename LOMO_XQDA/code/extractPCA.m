%Input rows are the sentences, cols are the variabls
function sentenceVectors=extractPCA(sentences, numDims)
    idx=randperm(size(sentences,2));
    newSize=10000;
    
    meanS=mean(sentences,2);%col vector
    sentenceVector=bsxfun(@minus,sentences,meanS);
    covy=sentenceVector(:,idx(1:newSize)).'*sentenceVector(:,idx(1:newSize))/size(sentenceVector,1);
    [eigVecs, eigVals]=eig(covy);%eigVecs cols are eigenVectors
    eigVals=diag(eigVals);%col vector
    eigVals=eigVals.';
    [eigVals,i]=sort(eigVals,'descend');
    eigVecs(:,:)=eigVecs(:,i);
    eigVecs=eigVecs(:,1:numDims);
    
    for s= 1: size(sentences,1);
        for i=1: numDims
            temp= sentenceVector(s,idx(1:newSize))*eigVecs(:,i);
            sentenceVectors(s,i)= temp;
        end
    end
    
end