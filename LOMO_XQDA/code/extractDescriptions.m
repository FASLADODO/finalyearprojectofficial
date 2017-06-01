%%Can only load sentences with same vector length
%sentence directory is '../../word2vec/trunk/matlab_sentence_vectors/';
%sentences saved and loaded as variables sss
%%  SAVE WHEN DATA EXTRACTED, SAVE INTERMEDIATE RUNTYPE 3 SENTENCES, 
%% CAN LOAD ALL AND EXTRACT CORRECTLY, CANT DO ALL ALONE, MUST DO ALL AND RUNTYPE, AS CANT LOAD CONTINGUOUSLY OTHERWISE
function [sentenceNames,sentences, sentenceIds]= extractDescriptions(sentenceDir, sentencesRun, sentencesRunType, preciseId, options)
    
    ids=table2cell(readtable('../../word2vec/trunk/imageIds.txt'));
    size(ids)
    ids(1,:)
    n=length(ids);
    sentenceIds=zeros(n,1);
    
    
    %n is number of sentence ids should be in order
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
    %% Convert sentenceIds to logical binary VERY IMPORTANT !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    fprintf('THere are %d sentenceIds', size(sentenceIds))
    
    
    sentenceList = dir([sentenceDir, '*.txt']);%[imgDir, '*.png']
    n = length(sentenceList);
    %if want all sentences extracted from directory
    if(strcmp(cellstr(sentencesRun(1)), 'all'))
        sentenceNames=sentenceList(:).name;
        %for every sentence config file
        for i= 1: n
            name=strcat(sentenceDir, sentenceList(i).name);
            %if sentences mode mean or max
            if(sentencesRunType~=3 && ~contrains(strfind(name,'norm3' )))
                temp=table2array(readtable(char(name),'Delimiter',' '));
                sentences(i,:,:)=temp(:,1:size(temp,2)-1);
            else
                    %if sentences represented as matrices
                    if(sentencesRunType==3 && contains(strfind(name,'norm3')))
                        temp=table2array(readtable(char(name),'Delimiter','comma'));
                        sentences(i,:,:)=temp(:,1:size(temp,2)-1);                
                    end
            end
            %sentenceNames(i)=sentenceList(i).name;
        end
    %If files loading sentences from are specified exactly
    else
        sentenceNames=sentencesRun;
        
        
        
        
        
        %for all files specified load sentences
        for i= 1: length(sentencesRun)
            name= strcat(sentenceDir,cellstr(sentencesRun(i)));
            %if type mean or mode sentences
            if(~contains(strfind(name,'norm3')) && sentencesRunType~=3 )
                temp=table2array(readtable(char(name),'Delimiter',' '));
                sentences(i,:,:)=temp(:,1:size(temp,2)-1);
            else
                    %if sentences are matrices
                    if(contains(strfind(name,'norm3')) && sentencesRunType==3)
                        %Prepare names to save sentences
                        name
                        saveSentences(i)=strrep(name,'.txt','');
                        saveSentences(i)=char(strrep(saveSentences(i),sentenceDir, ''))
                        saveSentences(i)= char(strcat( '../data/sentences/', saveSentences(i),  options.featureExtractionName, '_trainLevel', ...
                                                    options.trainLevel, '_',options.sentenceSplit, '_hiddensizes', ...
                                                    options.hiddensize1, options.hiddensize2,'_maxepochs',...
                                                    options.maxepoch1,options.maxepoch2,options.maxepoch3,'_trainsplit', ...
                                                    options.sentenceTrainSplit, '.mat'));

                        netSentences(i)=strrep(name,'.txt','');
                        netSentences(i)=strcat('../data/nets/', 

                        %convert sentences names to data held to check
                        %existence
                        storedSentences=strrep(name,'.txt','.mat');
                        storedSentences=char(strrep(storedSentences, sentenceDir, '../data/sentences/'))
                        size(storedSentences)
                        %if file exists
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
            %% Sentences are saved HERE
            extractionFunct=options.featureExtractionMethod;
            [sentences, sentenceIds]=extractionFunct(sentences, sentenceIds,options);
            
    end
    
end