%Sentences are 4D sentenceConfig, sentence, word, wordvector
%<images>: a set of n RGB color images. Size: [h, w, 3, n]
%gives LOMO descriptors. Size: [d, n] 
%gives features by no. sentences

% so want to give no.sentences, but multiply word vars by 3

function [sentences,sentenceIds]=convertToSentenceFeatures(sentences, sentenceIds, options)

    
    

    switch (options.featureExtractionMethod)
        case 'LOMO'
            newSentences=zeros(size(sentences,3), size(sentences,4),3,size(sentences,2));

            for config= 1: size(sentences,1)
                for s = 1:size(sentences,2)
                    size(newSentences(:,:,:,s));
                    %size([[squeeze(sentences(config,s,:,:))],[squeeze(sentences(config,s,:,:))],[squeeze(sentences(config,s,:,:))]])
                    %size([squeeze(sentences(config,s,:,:));squeeze(sentences(config,s,:,:));squeeze(sentences(config,s,:,:))])
                    size(cat(3, squeeze(sentences(config,s,:,:)),squeeze(sentences(config,s,:,:)),squeeze(sentences(config,s,:,:))));
                    %size([squeeze(sentences(config,s,:,:)).',squeeze(sentences(config,s,:,:)).',squeeze(sentences(config,s,:,:)).'])
                    %size([squeeze(sentences(config,s,:,:)).';squeeze(sentences(config,s,:,:)).';squeeze(sentences(config,s,:,:)).'])
                    newSentences(:,:,:,s)=cat(3, squeeze(sentences(config,s,:,:)),squeeze(sentences(config,s,:,:)),squeeze(sentences(config,s,:,:)));

                end
                 %[personIds, features]=featureFunct(images,person_ids   
                [sentenceIds2, sss]=LOMO(newSentences, sentenceIds,options);
                sentenceFeatures(config,:,:)=sss.';
            end
            sentences=sentenceFeatures;
            sentenceIds=sentenceIds2;
        case 'AUTOENCODE1'
            [sentences, sentenceIds]=autoEncodeSentences(sentences, sentenceIds,options);
        case 'AUTOENCODE2'
            [sentences, sentenceIds]=autoEncodeSentences(sentences, sentenceIds,options);
        case 'AUTOENCODE3'
            [sentences, sentenceIds]=autoEncodeSentences(sentences, sentenceIds,options);
end