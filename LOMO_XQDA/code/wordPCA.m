%similarityTests={'he', 'she', 'male', 'female', 'him','her', 'man','woman','boy','girl'};
similarityTests={'fat','large','slim','slender','phone','mobile','man','male','short','small','sneakers','shoes'};
similarityTests={'sneakers','shoes', 'trousers','jeans', 'male','man' 'shirt','t-shirt', 'bag','backpack','shoulder','bag', 'long','phone','small','files'};
%similarityTests={'long','short','long-sleeved','short-sleeved', 'old', 'young' 'large','small', 'tall', 'short'};

RELATION=0;
GROUP=1;
mode=GROUP;%RELATION
wordvectors= table2cell(readtable('../../word2vec/trunk/phrasevectors-txt/vectors-phrase-win5-threshold0-size500.txt'));
size(wordvectors);
words=wordvectors(:,1);
vectors=wordvectors(:,2:size(wordvectors,2)-1);
testvectors=zeros(numel(similarityTests),size(vectors,2));
for s= 1:numel(similarityTests)
    pos=find(strcmp(words,similarityTests{s}));
    for i=1:size(vectors,2)
        testvectors(s,i)=vectors{s,i}; 
    end
end
%rows are the full individ vectors, cols are the dims
labels=cell(numel(similarityTests)/2,1);
for i=1:length(labels)
   labels{i}= strcat(similarityTests{((i-1)*2)+1},'-', similarityTests{((i-1)*2)+2});
end

plotvectors=extractPCA(testvectors, 2);
figure
for i=1:2:size(plotvectors,1)
    plot(plotvectors(i,:),plotvectors(i+1,:))
    hold on
end
legend(labels);













