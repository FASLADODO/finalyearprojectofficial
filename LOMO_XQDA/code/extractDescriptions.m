%%Can only load sentences with same vector length
function [sentenceNames,sentences, sentenceIds]= extractDescriptions(sentenceDir, sentencesRun, preciseId, sentencesRunType, options)
    
    ids=table2cell(readtable('../../word2vec/trunk/imageIds.txt'));
    size(ids)
    ids(1,:)
    n=length(ids);
    sentenceIds=zeros(n,1);
    for i=1:n
       name=strjoin(ids(i,:));%Format image=06_set=3_id=0001
       if(preciseId)
           temp= strsplit(name,{'image ','_set ','_id ','.png'});
           sentenceIds(i)= str2double(strcat(temp(2),temp(3),temp(4)));%str2double()
       else
           temp= strsplit(name,{'image ','_set ','_id ','.png'});
           sentenceIds(i)= str2double(strcat(temp(3),temp(4)));%str2double()
       end
       %person_ids_char(i)= char(strcat(temp(3),temp(4)));
    end
    fprintf('THere are %d sentenceIds', size(sentenceIds))
    
    
    sentenceList = dir([sentenceDir, '*.txt']);%[imgDir, '*.png']
    n = length(sentenceList);
    
    if(strcmp(cellstr(sentencesRun(1)), 'all'))
        sentenceNames=sentenceList(:).name;
        for i= 1: n
            name=strcat(sentenceDir, sentenceList(i).name);
            if((sentencesRunType==1 | sentencesRunType==2) && isempty(strfind(name,'norm3')) )
                temp=table2array(readtable(char(name),'Delimiter',' '));
                sentences(i,:,:)=temp(:,1:size(temp,2)-1);
            else
                    if(sentencesRunType==3 && ~isempty(strfind(name,'norm3')))
                        temp=table2array(readtable(char(name),'Delimiter','comma'));
                        sentences(i,:,:)=temp(:,1:size(temp,2)-1);                
                    end
            end
            %sentenceNames(i)=sentenceList(i).name;
        end
    else
        sentenceNames=sentencesRun;
        for i= 1: length(sentencesRun)
            name= strcat(sentenceDir,cellstr(sentencesRun(i)));
            type3=strfind(name,'norm3');
            
            if((sentencesRunType==1 || sentencesRunType==2) && isempty(type3{1}) )
                temp=table2array(readtable(char(name),'Delimiter',' '));
                sentences(i,:,:)=temp(:,1:size(temp,2)-1);
            else
                    if(sentencesRunType==3 && ~isempty(type3{1}))
                        
                        storedSentences=strrep(name,'.txt','.mat');
                        storedSentences=char(strrep(storedSentences, sentenceDir, '../data/'))
                        size(storedSentences)
                        if exist(storedSentences, 'file') == 2
                            fprintf('Sentences %s already exists. Loading .mat', storedSentences);
                            load(storedSentences);
                            sentences(i,:,:,:)=sss;
                        else
                            fprintf('Loading sentences %s', char(name));
                            temp=table2cell(readtable(char(name),'Delimiter','comma'));
                            for st=1:size(temp,1)-1
                               for w=1:size(temp,2)-1
                                  %fprintf('%s \n',char(temp{st,w}));
                                  word=str2double(strsplit(char(temp{st,w}),' ')).';
                                  sentences(i,st,w,:)= word(1:size(word,1)-1);
                                  %fprintf('%d \n ',sentences(i,st,w,:));
                               end
                               fprintf('sentence %d\n',st);
                            end
                            sss=sentences(i,:,:,:);
                            size(sss)
                            save(storedSentences, 'sss');
                        end
                        %temp=table2array(readtable(char(name),'Delimiter','comma'));
                        %sentences(i,:,:)=temp(:,1:size(temp,2)-1);                
                    end
            end
        end
    end   
    if(sentencesRunType==3)
            %% If matrix sentences, convert to feature vectors
            [sentences,sentenceIds]=convertToSentenceFeatures(sentences, sentenceIds,options);
    end
    
end