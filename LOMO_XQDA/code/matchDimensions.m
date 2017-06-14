%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
%% matchDimensions. 
%  Used to prepare gal(sentence) and prob (img) features so that can be
%  compared. 
% first- takes first x elements, used to test neural network layout
% pca- takes most discriminative x elements of larger matrix
% nan_fill, extends smaller and fills with nans
% extend, extends smaller to match larger
% sentenceGalFea- featureextractor, sentence config, sentence, vector
% sentenceProbFea- featureextractor, sentence config, sentences, vector
% no sentences * 2 if general
function [sentenceGalFea, sentenceProbFea]=matchDimensions(sentenceProbFea,sentenceGalFea,sentenceProbIds, sentenceGalIds, dimensionMatchMethod, classifierName)
            fprintf('Size of sentenceprobFea %d %d and sentenceGalFea %d %d  pre matching\n',size(sentenceProbFea,3), size(sentenceProbFea,4),size(sentenceGalFea,3),size(sentenceGalFea,4)); 
            numDims=min(size(sentenceProbFea,4),size(sentenceGalFea,4));
            if(size(sentenceProbFea,4)  ~= size(sentenceGalFea,4))
                if(size(sentenceProbFea,4) > size(sentenceGalFea,4))
                    switch dimensionMatchMethod
                        case 'first'
                            sentenceProbFea=sentenceProbFea(:,:,:,1:numDims);
                        case 'pca'
                            for ft= 1:size(sentenceProbFea,1)
                                for c=1:size(sentenceProbFea,2)
                                    sentenceProbFea2(ft,c,:,:)= extractPCA(squeeze(sentenceProbFea(ft,c,:,:)),sentenceProbIds(ft,:), numDims);
                                end
                            end
                            sentenceProbFea=sentenceProbFea2;
			case 'lda'
                            for ft= 1:size(sentenceProbFea,1)
                                for c=1:size(sentenceProbFea,2)
                                    sentenceProbFea2(ft,c,:,:)= extractLDA(squeeze(sentenceProbFea(ft,c,:,:)),sentenceProbIds(ft,:),numDims);
                                end
                            end
                            sentenceProbFea=sentenceProbFea2;

                        case 'nan_fill'
                            tempNan= NaN(size(sentenceGalFea,1),size(sentenceGalFea,2),size(sentenceGalFea,3),size(sentenceProbFea,4)-size(sentenceGalFea,4));
                            
				sentenceGalFea(:,:,:,size(sentenceGalFea,4)+1:size(sentenceProbFea,4))=tempNan;
                        case 'extend'
                            for ft= 1:size(sentenceGalFea,1)
                                for c=1:size(sentenceGalFea,2)
                                    sentenceGalFea2(ft,c,:,:)= reshape(sentenceGalFea(ft,c,:,:),size(sentenceGalFea,3),size(sentenceProbFea,4));
                                end
                            end
                            sentenceGalFea=sentenceGalFea2;       
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
                        case 'nan_fill'

                            tempNan= NaN(size(sentenceProbFea,1),size(sentenceProbFea,2),size(sentenceProbFea,3),size(sentenceGalFea,4)-size(sentenceProbFea,4));
			
                            sentenceProbFea(:,:,:,size(sentenceProbFea,4)+1:size(sentenceGalFea,4))=tempNan;
			case 'lda'
			    
                            for ft= 1:size(sentenceGalFea,1)
                                for c=1:size(sentenceGalFea,2)
                                    sentenceGalFea2(ft,c,:,:)= extractLDA(squeeze(sentenceGalFea(ft,c,:,:)),sentenceGalIds(ft,:), numDims);
                                end
                            end
                            sentenceGalFea=sentenceGalFea2;
			    
                        case 'extend'
                            for ft= 1:size(sentenceProbFea,1)
                                for c=1:size(sentenceProbFea,2)
                                    sentenceProbFea2(ft,c,:,:)= reshape(sentenceProbFea(ft,c,:,:),size(sentenceProbFea,3),size(sentenceGalFea,4));
                                end
                            end
                            sentenceProbFea=sentenceProbFea2;     
                    end

                end
            end       
     fprintf('size sentenceGalFea %d %d and sentenceProbFea %d %d post matching\n',size(sentenceGalFea,3),size(sentenceGalFea,4),size(sentenceProbFea,3),size(sentenceProbFea,4))

end
