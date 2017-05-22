function [sentenceGalFea, sentenceProbFea]=matchDimensions(sentenceProbFea,sentenceGalFea, dimensionMatchMethod, classifierName)
    switch classifierName
        
        %%Make image and sentence vectors dimensions match for XQDA 
        case 'XQDA'
            numDims=min(size(sentenceProbFea,4),size(sentenceGalFea,4));
            if(size(sentenceProbFea,4)  ~= size(sentenceGalFea,4))
                if(size(sentenceProbFea,4) > size(sentenceGalFea,4))
                    switch dimensionMatchMethod
                        case 'first'
                            sentenceProbFea=sentenceProbFea(:,:,:,1:numDims);
                        case 'pca'
                            for ft= 1:size(sentenceProbFea,1)
                                for c=1:size(sentenceProbFea,2)
                                    sentenceProbFea2(ft,c,:,:)= extractPCA(squeeze(sentenceProbFea(ft,c,:,:)), numDims);
                                end
                            end
                            sentenceProbFea=sentenceProbFea2;        
                    end
                else
                   switch dimensionMatchMethod
                        case 'first'
                            sentenceGalFea=sentenceGalFea(:,:,:,1:numDims);
                        case 'pca'
                            for ft= 1:size(sentenceGalFea,1)
                                for c=1:size(sentenceGalFea,2)
                                    sentenceGalFea2(ft,c,:,:)= extractPCA(squeeze(sentenceGalFea(ft,c,:,:)), numDims);
                                end
                            end
                            sentenceGalFea=sentenceGalFea2;        
                    end

                end
            end
    end

end