%%Can only load sentences with same vector length
function [sentenceNames,sentences, sentenceIds]= extractDescriptions(sentenceDir, sentencesRun, preciseId)
    
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
    
    
    sentenceList = dir([sentenceDir, '*.txt']);%[imgDir, '*.png']
    n = length(sentenceList);
    
    if(strcmp(cellstr(sentencesRun(1)), 'all'))
        sentenceNames=sentenceList(:).name;
        for i= 1: n
            name=strcat(sentenceDir, sentenceList(i).name);
            temp=table2array(readtable(char(name),'Delimiter',' '));
            sentences(i,:,:)=temp(:,1:size(temp,2)-1);
            %sentenceNames(i)=sentenceList(i).name;
        end
    else
        sentenceNames=sentencesRun;
        for i= 1: length(sentencesRun)
            name= strcat(sentenceDir,cellstr(sentencesRun(i)));
            temp=table2array(readtable(char(name),'Delimiter',' '));
            sentences(i,:,:)=temp(:,1:size(temp,2)-1);
        end
    end   
    
end