in=readtable('../../CUHK03NLP.csv');
data=table2array(in(:,4));
imgs=table2array(in(:,2));
%data = csvread('../../CUHK03NLP.csv',1,0);
%[s,data,r] = xlsread('../../CUHK03NLP.csv');
fileID = fopen('../../word2vec/trunk/descriptions.txt','w');
imageID= fopen('../../word2vec/trunk/imageIds.txt','w');
for i=2:size(data,1) 
    if(~strcmp(char(data(i)),'none'))
        fprintf(imageID,'%s\n',char(imgs(i)));
        fprintf(fileID,'%s\n',char(data(i)));
    end
end
fclose(fileID);
fclose(imageID);
