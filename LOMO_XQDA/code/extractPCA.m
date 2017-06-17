%Input rows are the sentences, cols are the variabls
function sentenceVectors=extractPCA(sentences, numDims)
    idx=randperm(size(sentences,2));
    newSize=min(10000,size(sentences,2));%Newsize necessary to reduce memory usage
    sentences=sentences(:,idx(1:newSize));
    meanS=mean(sentences,2);%col vector
    sentenceVector=bsxfun(@minus,sentences,meanS);
    covy=sentenceVector(:,:).'*sentenceVector(:,:)/size(sentenceVector,1);
    [eigVecs, eigVals]=eig(covy);%eigVecs cols are eigenVectors
    eigVals=diag(eigVals);%col vector
    eigVals=eigVals.';
    [eigVals,i]=sort(eigVals,'descend');
    eigVecs(:,:)=eigVecs(:,i);
    eigVecs=eigVecs(:,1:numDims);
    
    for s= 1: size(sentences,1)
        for i=1: numDims
            temp= sentenceVector(s,:)*eigVecs(:,i);
            sentenceVectors(s,i)= temp;
        end
    end
    
end