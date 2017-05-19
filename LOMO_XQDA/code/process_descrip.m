%file=fopen('../../word2vec/trunk/imageIds.txt','r');
%Imgs=fread(file);
%phraseFile=fopen('../../word2vec/trunk/word-vects.txt','r');
%Phrases=fread(phraseFile);
imgIds=readtable('../../word2vec/trunk/imageIds.txt','Delimiter',' ');
%phrases=readtable('../../word2vec/trunk/word-vects.txt','Delimiter',' ');
sentenceVectorDir='../../word2vec/trunk/matlab_sentence_vectors/';



sentenceList = dir([sentenceVectorDir, '*.txt']);
n = length(sentenceList);
vecLength=200;
noSentences=size(imgIds,1);


%% Allocate memory
sentences = zeros(noSentences,vecLength, n);
sentenceIds= cell(n,1);

%% read sentences
for i = 1 : n
    sentenceIds(i)=cellstr(sentenceList(i).name);
    name=strcat(sentenceVectorDir,sentenceList(i).name);
    curr=readtable(name,'Delimiter',' ');
    sentences(:,:,i)=table2array(curr(:,1:200));
end

