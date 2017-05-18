%file=fopen('../../word2vec/trunk/imageIds.txt','r');
%Imgs=fread(file);
%phraseFile=fopen('../../word2vec/trunk/word-vects.txt','r');
%Phrases=fread(phraseFile);
imgs=readtable('../../word2vec/trunk/imageIds.txt','Delimiter',' ');
phrases=readtable('../../word2vec/trunk/word-vects.txt','Delimiter',' ');