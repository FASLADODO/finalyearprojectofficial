function [newIds]= hotCoding(sentenceIds)
%function [newIds, newSentences]= hotcoding(sentenceIds, sentences)

    numberIds= length(unique(sentenceIds));
    [orderedIds, idx]= sort(sentenceIds);
    %newSentences= sentences(:,idx,:,:);
    temp=zeros(1, numberIds);
    powow=1;
    oldVal=orderedIds(1);
    newIds=zeros(length(sentenceIds),numberIds);
    for i=1:length(sentenceIds)+1
        if(i<=length(sentenceIds))
            if(oldVal~=orderedIds(i))
               %'oldval not equal orderedIds(i)'
               for u=4:-1:1
                   if(i-u>0)
                       %'I greater than u'
                       if(orderedIds(i-u)==oldVal)
                           addedVal=de2bi(temp);  
                           addedVal(powow)=1;
                          % addedVal=de2bi(temp)+de2bi((2).^powow)
                          %addedVal;
                          %numberIds;
                          %size(newIds)
                           newIds(i-u,:)=addedVal.';
                       end
                   end
               end
               powow=powow+1;
               oldVal=orderedIds(i);
            end 
        else
            for u=4:-1:1
                   if(i-u>0)
                       'I greater than u';
                       if(orderedIds(i-u)==oldVal)
                           %{
                           addedVal=de2bi(temp);
                           tempy=de2bi((2).^powow);
                           for p=1:length(tempy)
                              addedVal(p)=addedVal(p)+tempy(p); 
                           end
                           %}
                           addedVal=de2bi(temp);  
                           addedVal(powow)=1;                          
                          % addedVal=de2bi(temp)+de2bi((2).^powow)
                          %addedVal
                          %numberIds
                          %size(newIds)
                           newIds(i-u,:)=addedVal.';
                       end
                   end
            end
        end
    end
    newIds(idx,:)=newIds;
    %finalIds=cell(length(sentenceIds),1);
   % finalIds=zeros(length(sentenceIds),length(unique(sentenceIds)));
    %for i =1:length(sentenceIds)
    %    finalIds(i,:)=newIds(i,:);%mat2cell(newIds(i,:),1)
   % end
end