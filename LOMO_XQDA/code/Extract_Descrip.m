%data = csvread('../../CUHK03NLP.csv');
[s,data,r] = xlsread('../../CUHK03NLP.csv');
fileID = fopen('../../descriptions.txt','w');
for i=2:size(data,1)    
    fprintf(fileID,'%s\n',char(data(i,4)));
end
fclose(fileID);
