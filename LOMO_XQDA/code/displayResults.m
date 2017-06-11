%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%displayResults.m
%%Method to show bunch of results together if already exist
%%Will NOT extract new ones
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%5
clear; clc;
close all;

IMAGES=0;
SENTENCES=1;
SENTENCEIMAGES=2;

NORMAL=0;
SIMWORDS=1;
USELESSWORDS=2;

MEAN=1;
MODE=2;
MATRIX=3;

READ_ALL=3;
READ_DISTORT=4;

mode=IMAGES;

imageClassifier='XQDA';
imageFeature='autoEncode2d';
trainLevel=3;
noImages=0;
imageTrainType='pairs';
imageSizes=[50];%display all imageHiddenSizes default
imageRetinex=[0]; %1 or 0
imageMaxEpoch=[[100,50,100]];%[200,100,200], [400,200,400]
imageReadIn=[READ_DISTORT];%READ_ALL
imageTrainingSizes= [1000]; %2000,1500,1000,500
imageHiddenSizes=[[500,200];[500,100];[1000,200];[1000,100];[800,200];[800,100];[500,200];[500,100];[1500,200];[1500,100]];
imageNumRanks=1000;

sentenceSizes=[100,200,300,400,500];
sentenceNormalisation=[MATRIX];%MEAN, MODE
sentenceModes=[NORMAL];%,SIMWORDS,USELESSWORDS
sentenceWindows=[3,5,7,10];
sentenceThresholds=[200,150,0];

colormap winter
figure;
switch(mode)
    case IMAGES
        %% Create given configuration for all graphs shown
        config='';
        vals=[imageSizes(1),imageRetinex(1),[imageMaxEpoch(1)],imageReadIn(1),imageTrainingSizes(1), [imageHiddenSizes(1)]];
        descrips={'imagesize ','retinex ','maxepochs ', 'imagereadinmethod ','imagetrainingsize ', 'hiddensizes '};
        dimensions=[length(imageSizes),length(imageRetinex),size(imageMaxEpoch,1),length(imageReadIn),length(imageTrainingSizes), size(imageHiddenSizes,1)];
        idx=find(dimensions==1);
        varyIndexes=setdiff([1:size(dimensions,2)],idx);
        for i=1:length(idx)
           %descrips(idx(i))
           config=strcat(config,descrips(idx(i)),num2str(vals(idx(i))));
        end
        labels=cell(length(imageSizes)*length(imageRetinex)*length(imageReadIn)*length(imageTrainingSizes)*size(imageMaxEpoch,1)*size(imageHiddenSizes,1),1);
        %% Find if images have been extracted and show results
        for s=1:length(imageSizes)
           for r= 1:length(imageRetinex)
               for i=1:length(imageReadIn)
                  for t=1:length(imageTrainingSizes)
                      for e=1:size(imageMaxEpoch,1)
                          for h=1:size(imageHiddenSizes,1)
                              %Get variedparts
                              name='';
                              for n= 1: length(varyIndexes)
                                  switch (varyIndexes(n))
                                      case 1
                                          name=strcat(name,'imageSize-',num2str(imageSizes(s)));
                                      case 2
                                          name=strcat(name,'retinex-',num2str(imageRetinex(r)));
                                      case 4
                                          name=strcat(name,'imagereadin method-',num2str(imageReadIn(i)));
                                      case 5
                                          name=strcat(name,'trainingSize-',num2str(imageTrainingSizes(t)));
                                      case 3
                                          name=strcat(name,'maxEpochs-',num2str(imageMaxEpoch(e,1)),num2str(imageMaxEpoch(e,2)),num2str(imageMaxEpoch(e,3)));
                                      case 6
                                          name=strcat(name,'hiddenSizes-',num2str(imageHiddenSizes(h,1)),num2str(imageHiddenSizes(h,2)));
                                  end
                              end
                              fileNames={imageClassifier,imageFeature,num2str(noImages),num2str(imageReadIn(i)), num2str(imageTrainingSizes(t)), imageTrainType,num2str(trainLevel),num2str( imageHiddenSizes(h,1)), num2str(imageHiddenSizes(h,2)), num2str(imageMaxEpoch(e,1)),num2str(imageMaxEpoch(e,2)),num2str(imageMaxEpoch(e,3)),num2str(imageRetinex(r)),num2str(imageSizes(s)),num2str(imageSizes(s))};
                              
                              fileName=strjoin(fileNames,'-');
                              fileName=strcat('../results/images/',fileName,'.mat');
                                if (exist(fileName, 'file') == 2) 
                                    fprintf('filename %s found, extracting now\n',fileName);
                                    labels{h+(size(imageHiddenSizes,1)*(e-1))+(size(imageHiddenSizes,1)*size(imageMaxEpoch,1)*(t-1))+(size(imageHiddenSizes,1)*length(imageTrainingSizes)*size(imageMaxEpoch,1)*(i-1))+(size(imageHiddenSizes,1)*length(imageTrainingSizes)*size(imageMaxEpoch,1)*length(imageReadIn)*(r-1))+(size(imageHiddenSizes,1)*length(imageTrainingSizes)*size(imageMaxEpoch,1)*length(imageReadIn)*length(imageRetinex)*(s-1)),1}=name; 
                                    load( fileName);
                                    plot(1 : size(meanCms,2), meanCms,'LineWidth',1.5)
                                    hold on;
                                else
                                    fprintf('filename %s not found\n',fileName);
                                end
                          end
                      end
                  end
               end
           end
        end
        labels
        title(sprintf('CMS Curve for CUHK03 Image Matching with %s', config{1}))
        xlabel('No. Ranks of ordered Gallery Images') % x-axis label
        ylabel('% Gallery Images that contain match within that rank') % y-axis label
        labels=labels(~cellfun('isempty',labels))  
        legend(labels);
    case SENTENCES
        
    case SENTENCEIMAGES
        
    
    
end


    
 