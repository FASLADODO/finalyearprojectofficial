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
SENTENCESPECIFIC=3;
IMAGESPECIFIC=4;
NORMAL=0;
SIMWORDS=1;
USELESSWORDS=2;

MEAN=1;
MAX=2;
MATRIX=3;

READ_ALL=3;
READ_DISTORT=4;

mode=IMAGESPECIFIC;

imageClassifier='XQDA';
imageFeature='autoEncode2d';
trainLevel=3;
noImages=0;
imageTrainType='pairs';
imageSizes=[40,50];%display all imageHiddenSizes default
imageRetinex=[0]; %1 or 0
imageMaxEpoch=[[100,50,100]];%[200,100,200], [400,200,400]
imageReadIn=[READ_DISTORT];%READ_ALL
imageTrainingSizes= [1000,2000]; %2000,1500,1000,500
imageHiddenSizes=[[1000,200];[1500, 200];[1500,100]];
imageNumRanks=1000;

% Will be the same for all exps
sentenceFeature='autoEncodeSentences';
sentenceTrainLevel='3';
sentenceArrange='pairs';
sentencePrecise=0;
% Will vary
sentenceClassifier={'XQDA'}; %twoChannel2, twoChannel3, XQDA
sentenceSizes=[100,200,300,400,500];
sentenceNormalisation=[MAX];%MEAN, MODE
sentenceModes=[USELESSWORDS, SIMWORDS, NORMAL];%,SIMWORDS,USELESSWORDS
sentenceWindows=[3,5,7,10];
sentenceThresholds=[200,150,0];
sentenceHiddenSizes=[[100,40]];%etc
sentenceMaxEpochs=[[20,10,100]];
sentenceTrainSplit=[2000];%200
sentenceTop=0.4;%0 by default
sentencesRun={
   %'XQDA_mode2_norm3outvectors_phrase_win7_threshold0_size100autoEncodeSentences_trainLevel3_pairs_hiddensizes10040_maxepochs2010100_trainsplit2000_precise0',...
  % 'XQDA_mode2_norm3outvectors_phrase_win10_threshold200_size200autoEncodeSentences_trainLevel3_pairs_hiddensizes10040_maxepochs2010100_trainsplit2000_precise0',...
   %'XQDA_mode2_norm3outvectors_phrase_win3_threshold200_size300autoEncodeSentences_trainLevel3_pairs_hiddensizes10040_maxepochs2010100_trainsplit2000_precise0',...
  % 'XQDA_mode2_norm3outvectors_phrase_win10_threshold200_size400autoEncodeSentences_trainLevel3_pairs_hiddensizes10040_maxepochs2010100_trainsplit2000_precise0',...
  % 'XQDA_mode2_norm3outvectors_phrase_win10_threshold200_size500autoEncodeSentences_trainLevel3_pairs_hiddensizes10040_maxepochs2010100_trainsplit2000_precise0'
   %'XQDA_mode1_norm3outvectors_phrase_win5_threshold0_size100autoEncodeSentences_trainLevel3_pairs_hiddensizes10040_maxepochs2010100_trainsplit2000_precise0',...
  % 'XQDA_mode1_norm3outvectors_phrase_win3_threshold150_size200autoEncodeSentences_trainLevel3_pairs_hiddensizes10040_maxepochs2010100_trainsplit2000_precise0',...
  % 'XQDA_mode1_norm3outvectors_phrase_win5_threshold0_size300autoEncodeSentences_trainLevel3_pairs_hiddensizes10040_maxepochs2010100_trainsplit2000_precise0',...
  % 'XQDA_mode1_norm3outvectors_phrase_win5_threshold0_size400autoEncodeSentences_trainLevel3_pairs_hiddensizes10040_maxepochs2010100_trainsplit2000_precise0',...
  % 'XQDA_mode1_norm3outvectors_phrase_win5_threshold200_size500autoEncodeSentences_trainLevel3_pairs_hiddensizes10040_maxepochs2010100_trainsplit2000_precise0'
    'XQDA_mode2_norm3outvectors_phrase_win10_threshold200_size200autoEncodeSentences_trainLevel3_pairs_hiddensizes10040_maxepochs2010100_trainsplit2000_precise0',...
    'XQDA_mode1_norm3outvectors_phrase_win5_threshold0_size100autoEncodeSentences_trainLevel3_pairs_hiddensizes10040_maxepochs2010100_trainsplit2000_precise0',...
    'XQDA_mode0_norm3outvectors_phrase_win5_threshold200_size200autoEncodeSentences_trainLevel3_pairs_hiddensizes10040_maxepochs2010100_trainsplit2000_precise0',...
    'XQDA_mode0_norm3outvectors_phrase_win10_threshold0_size300autoEncodeSentences_trainLevel3_pairs_hiddensizes10040_maxepochs2010100_trainsplit2000_precise0',...
    'XQDA_mode0_norm3outvectors_phrase_win7_threshold0_size400autoEncodeSentences_trainLevel3_pairs_hiddensizes10040_maxepochs2010100_trainsplit2000_precise0',...
    'XQDA_mode0_norm3outvectors_phrase_win5_threshold150_size400autoEncodeSentences_trainLevel3_pairs_hiddensizes10040_maxepochs2010100_trainsplit2000_precise0',...
    'XQDA_mode0_norm3outvectors_phrase_win3_threshold0_size500autoEncodeSentences_trainLevel3_pairs_hiddensizes10040_maxepochs2010100_trainsplit2000_precise0'
   
   
   
    %'autoEncodeMatches_mode0_norm3outvectors_phrase_win10_threshold200_size300autoEncodeSentences_trainLevel3_pairs_hiddensizes10040_maxepochs2010100_trainsplit2000_precise0.mat',...
    %'twoChannel2_mode1_norm3outvectors_phrase_win10_threshold200_size500autoEncodeSentences_trainLevel3_pairs_hiddensizes10040_maxepochs2010100_trainsplit2000_precise0',...
    %'twoChannel_mode0_norm3outvectors_phrase_win3_threshold100_size50autoEncodeSentences_trainLevel3_pairs_hiddensizes200100_maxepochs102050_trainsplit200.mat'
};

sentenceImagesRun={
   % 'twoChannel2_autoEncode2d_1pca200402010201010051mode0_norm3outvectors_phrase_win10_threshold150_size100autoEncodeSentences_trainLevel3_pairs_hiddensizes10040_maxepochs2010100_trainsplit2000_precise0.mat.mat',...
    %'twoChannel2_autoEncode2d_1pca2004020102010100510.01500mode1_norm3outvectors_phrase_win5_threshold0_size100autoEncodeSentences_trainLevel3_pairs_hiddensizes10040_maxepochs2010100_trainsplit2000_precise0.mat.mat',...
    %'twoChannel3_autoEncode2d_1pca2004020102010100510.01500mode0_norm3outvectors_phrase_win10_threshold150_size100autoEncodeSentences_trainLevel3_pairs_hiddensizes10040_maxepochs2010100_trainsplit2000_precise0.mat.mat'

%    'precisetwoChannel2_autoEncode2d_1lda1004020102010100500.0110001mode1_norm3outvectors_phrase_win5_threshold0_size100autoEncodeSentences_trainLevel3_pairs_hiddensizes10040_maxepochs2010100_trainsplit2000_precise0.mat.mat',... 
%    'precisetwoChannel2_autoEncode2d_1pca1004020102010100500.0110001mode1_norm3outvectors_phrase_win5_threshold0_size100autoEncodeSentences_trainLevel3_pairs_hiddensizes10040_maxepochs2010100_trainsplit2000_precise0.mat.mat',... 
%              'twoChannel2_autoEncode2d_1lda1004020102010100500.0110000mode1_norm3outvectors_phrase_win5_threshold0_size100autoEncodeSentences_trainLevel3_pairs_hiddensizes10040_maxepochs2010100_trainsplit2000_precise0.mat.mat', ...
%               'twoChannel2_autoEncode2d_1pca1004020102010100500.0110000mode1_norm3outvectors_phrase_win5_threshold0_size100autoEncodeSentences_trainLevel3_pairs_hiddensizes10040_maxepochs2010100_trainsplit2000_precise0.mat.mat' ,...
% 'twoChannel2_LOMO_4lda1004020102010100500.0310000mode1_norm3outvectors_phrase_win5_threshold0_size100autoEncodeSentences_trainLevel3_pairs_hiddensizes10040_maxepochs2010100_trainsplit2000_precise0.mat.mat',...
%'twoChannel2_LOMO_10lda1004020102010100500.0310000mode1_norm3outvectors_phrase_win5_threshold0_size100autoEncodeSentences_trainLevel3_pairs_hiddensizes10040_maxepochs2010100_trainsplit2000_precise0.mat.mat',...
%              'twoChannel2_LOMO_4lda1004020102010100500.0210000mode1_norm3outvectors_phrase_win5_threshold0_size100autoEncodeSentences_trainLevel3_pairs_hiddensizes10040_maxepochs2010100_trainsplit2000_precise0.mat.mat',...
%'twoChannel2_LOMO_10lda1004020102010100500.0210000mode1_norm3outvectors_phrase_win5_threshold0_size100autoEncodeSentences_trainLevel3_pairs_hiddensizes10040_maxepochs2010100_trainsplit2000_precise0.mat.mat',...
%'precisetwoChannel2_LOMO_1lda1004020102010100500.0110001mode1_norm3outvectors_phrase_win5_threshold0_size100autoEncodeSentences_trainLevel3_pairs_hiddensizes10040_maxepochs2010100_trainsplit2000_precise0.mat.mat',...
%'twoChannel2_LOMO_1lda1004020102010100500.0110000mode1_norm3outvectors_phrase_win5_threshold0_size100autoEncodeSentences_trainLevel3_pairs_hiddensizes10040_maxepochs2010100_trainsplit2000_precise0.mat.mat',...
%'twoChannel2_LOMO_2lda1004020102010100500.0110000mode1_norm3outvectors_phrase_win5_threshold0_size100autoEncodeSentences_trainLevel3_pairs_hiddensizes10040_maxepochs2010100_trainsplit2000_precise0.mat.mat',...
%'twoChannel2_LOMO_3lda1004020102010100500.0110000mode1_norm3outvectors_phrase_win5_threshold0_size100autoEncodeSentences_trainLevel3_pairs_hiddensizes10040_maxepochs2010100_trainsplit2000_precise0.mat.mat',...
%'twoChannel2_LOMO_4lda1004020102010100500.0110000mode1_norm3outvectors_phrase_win5_threshold0_size100autoEncodeSentences_trainLevel3_pairs_hiddensizes10040_maxepochs2010100_trainsplit2000_precise0.mat.mat',...
%'twoChannel2_LOMO_5lda1004020102010100500.0110000mode1_norm3outvectors_phrase_win5_threshold0_size100autoEncodeSentences_trainLevel3_pairs_hiddensizes10040_maxepochs2010100_trainsplit2000_precise0.mat.mat',...
%'twoChannel2_LOMO_10lda1004020102010100500.0110000mode1_norm3outvectors_phrase_win5_threshold0_size100autoEncodeSentences_trainLevel3_pairs_hiddensizes10040_maxepochs2010100_trainsplit2000_precise0.mat.mat'

%'twoChannel2_autoEncode2d_mirrored1lda1004020102010100500.0110000mode1_norm3outvectors_phrase_win5_threshold0_size100autoEncodeSentences_trainLevel3_pairs_hiddensizes10040_maxepochs2010100_trainsplit2000_precise0.mat.mat',...
%'twoChannel2_autoEncode2d_mirrored1pca1004020102010100500.0110000mode1_norm3outvectors_phrase_win5_threshold0_size100autoEncodeSentences_trainLevel3_pairs_hiddensizes10040_maxepochs2010100_trainsplit2000_precise0.mat.mat',...
%'twoChannel2_autoEncode2d_none1lda1004020102010100500.0110000mode1_norm3outvectors_phrase_win5_threshold0_size100autoEncodeSentences_trainLevel3_pairs_hiddensizes10040_maxepochs2010100_trainsplit2000_precise0.mat.mat',...
%'twoChannel2_autoEncode2d_none1pca1004020102010100500.0110000mode1_norm3outvectors_phrase_win5_threshold0_size100autoEncodeSentences_trainLevel3_pairs_hiddensizes10040_maxepochs2010100_trainsplit2000_precise0.mat.mat',...
%'twoChannel2_LOMO_mirrored1lda1004020102010100500.0110000mode1_norm3outvectors_phrase_win5_threshold0_size100autoEncodeSentences_trainLevel3_pairs_hiddensizes10040_maxepochs2010100_trainsplit2000_precise0.mat.mat',...
%'twoChannel2_LOMO_mirrored1pca1004020102010100500.0110000mode1_norm3outvectors_phrase_win5_threshold0_size100autoEncodeSentences_trainLevel3_pairs_hiddensizes10040_maxepochs2010100_trainsplit2000_precise0.mat.mat',...
%'twoChannel2_LOMO_none1lda1004020102010100500.0110000mode1_norm3outvectors_phrase_win5_threshold0_size100autoEncodeSentences_trainLevel3_pairs_hiddensizes10040_maxepochs2010100_trainsplit2000_precise0.mat.mat',...
%'twoChannel2_LOMO_none1pca1004020102010100500.0110000mode1_norm3outvectors_phrase_win5_threshold0_size100autoEncodeSentences_trainLevel3_pairs_hiddensizes10040_maxepochs2010100_trainsplit2000_precise0.mat.mat',...
%'precisetwoChannel2_autoEncode2d_mirrored1lda1004020102010100500.0110001mode1_norm3outvectors_phrase_win5_threshold0_size100autoEncodeSentences_trainLevel3_pairs_hiddensizes10040_maxepochs2010100_trainsplit2000_precise0.mat.mat',...
%'precisetwoChannel2_autoEncode2d_mirrored1pca1004020102010100500.0110001mode1_norm3outvectors_phrase_win5_threshold0_size100autoEncodeSentences_trainLevel3_pairs_hiddensizes10040_maxepochs2010100_trainsplit2000_precise0.mat.mat',...
%'precisetwoChannel2_autoEncode2d_none1lda1004020102010100500.0110001mode1_norm3outvectors_phrase_win5_threshold0_size100autoEncodeSentences_trainLevel3_pairs_hiddensizes10040_maxepochs2010100_trainsplit2000_precise0.mat.mat',...
%'precisetwoChannel2_autoEncode2d_none1pca1004020102010100500.0110001mode1_norm3outvectors_phrase_win5_threshold0_size100autoEncodeSentences_trainLevel3_pairs_hiddensizes10040_maxepochs2010100_trainsplit2000_precise0.mat.mat',...
%'precisetwoChannel2_LOMO_none1lda1004020102010100500.0310001mode1_norm3outvectors_phrase_win5_threshold0_size100autoEncodeSentences_trainLevel3_pairs_hiddensizes10040_maxepochs2010100_trainsplit2000_precise0.mat.mat',...
%'precisetwoChannel2_LOMO_mirrored1lda1004020102010100500.0310001mode1_norm3outvectors_phrase_win5_threshold0_size100autoEncodeSentences_trainLevel3_pairs_hiddensizes10040_maxepochs2010100_trainsplit2000_precise0.mat.mat',...
%'twoChannel2_LOMO_none1lda1004020102010100500.0310000mode1_norm3outvectors_phrase_win5_threshold0_size100autoEncodeSentences_trainLevel3_pairs_hiddensizes10040_maxepochs2010100_trainsplit2000_precise0.mat.mat',...
%'twoChannel2_LOMO_mirrored1lda1004020102010100500.0310000mode1_norm3outvectors_phrase_win5_threshold0_size100autoEncodeSentences_trainLevel3_pairs_hiddensizes10040_maxepochs2010100_trainsplit2000_precise0.mat.mat',...

%RESULTS FOR AUTOENCODE
%'precisetwoChannel2_autoEncode2d_mirrored1lda1004020102010100500.0110001mode1_norm3outvectors_phrase_win5_threshold0_size100autoEncodeSentences_trainLevel3_pairs_hiddensizes10040_maxepochs2010100_trainsplit2000_precise0.mat.mat',...
%'precisetwoChannel2_autoEncode2d_mirrored1pca1004020102010100500.0110001mode1_norm3outvectors_phrase_win5_threshold0_size100autoEncodeSentences_trainLevel3_pairs_hiddensizes10040_maxepochs2010100_trainsplit2000_precise0.mat.mat',...
%'precisetwoChannel2_autoEncode2d_none1lda1004020102010100500.0110001mode1_norm3outvectors_phrase_win5_threshold0_size100autoEncodeSentences_trainLevel3_pairs_hiddensizes10040_maxepochs2010100_trainsplit2000_precise0.mat.mat',...
%'precisetwoChannel2_autoEncode2d_none1pca1004020102010100500.0110001mode1_norm3outvectors_phrase_win5_threshold0_size100autoEncodeSentences_trainLevel3_pairs_hiddensizes10040_maxepochs2010100_trainsplit2000_precise0.mat.mat',...
%'twoChannel2_autoEncode2d_mirrored1lda1004020102010100500.0110000mode1_norm3outvectors_phrase_win5_threshold0_size100autoEncodeSentences_trainLevel3_pairs_hiddensizes10040_maxepochs2010100_trainsplit2000_precise0.mat.mat',...
%'twoChannel2_autoEncode2d_mirrored1pca1004020102010100500.0110000mode1_norm3outvectors_phrase_win5_threshold0_size100autoEncodeSentences_trainLevel3_pairs_hiddensizes10040_maxepochs2010100_trainsplit2000_precise0.mat.mat',...
%'twoChannel2_autoEncode2d_none1lda1004020102010100500.0110000mode1_norm3outvectors_phrase_win5_threshold0_size100autoEncodeSentences_trainLevel3_pairs_hiddensizes10040_maxepochs2010100_trainsplit2000_precise0.mat.mat',...
%'twoChannel2_autoEncode2d_none1pca1004020102010100500.0110000mode1_norm3outvectors_phrase_win5_threshold0_size100autoEncodeSentences_trainLevel3_pairs_hiddensizes10040_maxepochs2010100_trainsplit2000_precise0.mat.mat',...

%RESULTS FOR LOMO
%'precisetwoChannel2_LOMO_mirrored1lda1004020102010100500.0110001mode1_norm3outvectors_phrase_win5_threshold0_size100autoEncodeSentences_trainLevel3_pairs_hiddensizes10040_maxepochs2010100_trainsplit2000_precise0.mat.mat',...
%'precisetwoChannel2_LOMO_mirrored1pca1004020102010100500.0110001mode1_norm3outvectors_phrase_win5_threshold0_size100autoEncodeSentences_trainLevel3_pairs_hiddensizes10040_maxepochs2010100_trainsplit2000_precise0.mat.mat',...
%'precisetwoChannel2_LOMO_none1lda1004020102010100500.0110001mode1_norm3outvectors_phrase_win5_threshold0_size100autoEncodeSentences_trainLevel3_pairs_hiddensizes10040_maxepochs2010100_trainsplit2000_precise0.mat.mat',...
%'precisetwoChannel2_LOMO_none1pca1004020102010100500.0110001mode1_norm3outvectors_phrase_win5_threshold0_size100autoEncodeSentences_trainLevel3_pairs_hiddensizes10040_maxepochs2010100_trainsplit2000_precise0.mat.mat',...
%'twoChannel2_LOMO_mirrored1lda1004020102010100500.0110000mode1_norm3outvectors_phrase_win5_threshold0_size100autoEncodeSentences_trainLevel3_pairs_hiddensizes10040_maxepochs2010100_trainsplit2000_precise0.mat.mat',...
%'twoChannel2_LOMO_mirrored1pca1004020102010100500.0110000mode1_norm3outvectors_phrase_win5_threshold0_size100autoEncodeSentences_trainLevel3_pairs_hiddensizes10040_maxepochs2010100_trainsplit2000_precise0.mat.mat',...
%'twoChannel2_LOMO_none1lda1004020102010100500.0110000mode1_norm3outvectors_phrase_win5_threshold0_size100autoEncodeSentences_trainLevel3_pairs_hiddensizes10040_maxepochs2010100_trainsplit2000_precise0.mat.mat',...
%'twoChannel2_LOMO_none1pca1004020102010100500.0110000mode1_norm3outvectors_phrase_win5_threshold0_size100autoEncodeSentences_trainLevel3_pairs_hiddensizes10040_maxepochs2010100_trainsplit2000_precise0.mat.mat'

%RESULTS FOR BEST LOMO (MIRRORED, LDA, NON-PRECISE) DIFF FALSEPOS RATIOS
%'twoChannel2_LOMO_mirrored1lda1004020102010100500.0110000mode1_norm3outvectors_phrase_win5_threshold0_size100autoEncodeSentences_trainLevel3_pairs_hiddensizes10040_maxepochs2010100_trainsplit2000_precise0.mat.mat',...
%'twoChannel2_LOMO_2lda1004020102010100500.0110000mode1_norm3outvectors_phrase_win5_threshold0_size100autoEncodeSentences_trainLevel3_pairs_hiddensizes10040_maxepochs2010100_trainsplit2000_precise0.mat.mat',...
%'twoChannel2_LOMO_3lda1004020102010100500.0110000mode1_norm3outvectors_phrase_win5_threshold0_size100autoEncodeSentences_trainLevel3_pairs_hiddensizes10040_maxepochs2010100_trainsplit2000_precise0.mat.mat',...
%'twoChannel2_LOMO_4lda1004020102010100500.0110000mode1_norm3outvectors_phrase_win5_threshold0_size100autoEncodeSentences_trainLevel3_pairs_hiddensizes10040_maxepochs2010100_trainsplit2000_precise0.mat.mat',...
%'twoChannel2_LOMO_5lda1004020102010100500.0110000mode1_norm3outvectors_phrase_win5_threshold0_size100autoEncodeSentences_trainLevel3_pairs_hiddensizes10040_maxepochs2010100_trainsplit2000_precise0.mat.mat',...
%'twoChannel2_LOMO_10lda1004020102010100500.0110000mode1_norm3outvectors_phrase_win5_threshold0_size100autoEncodeSentences_trainLevel3_pairs_hiddensizes10040_maxepochs2010100_trainsplit2000_precise0.mat.mat'

%two channel learning rates
%'twoChannel2_LOMO_4lda1004020102010100500.0110000mode1_norm3outvectors_phrase_win5_threshold0_size100autoEncodeSentences_trainLevel3_pairs_hiddensizes10040_maxepochs2010100_trainsplit2000_precise0.mat.mat',...
%'twoChannel2_LOMO_4lda1004020102010100500.0210000mode1_norm3outvectors_phrase_win5_threshold0_size100autoEncodeSentences_trainLevel3_pairs_hiddensizes10040_maxepochs2010100_trainsplit2000_precise0.mat.mat',...
%'twoChannel2_LOMO_4lda1004020102010100500.0310000mode1_norm3outvectors_phrase_win5_threshold0_size100autoEncodeSentences_trainLevel3_pairs_hiddensizes10040_maxepochs2010100_trainsplit2000_precise0.mat.mat',...
%'twoChannel2_LOMO_10lda1004020102010100500.0110000mode1_norm3outvectors_phrase_win5_threshold0_size100autoEncodeSentences_trainLevel3_pairs_hiddensizes10040_maxepochs2010100_trainsplit2000_precise0.mat.mat',...
%'twoChannel2_LOMO_10lda1004020102010100500.0210000mode1_norm3outvectors_phrase_win5_threshold0_size100autoEncodeSentences_trainLevel3_pairs_hiddensizes10040_maxepochs2010100_trainsplit2000_precise0.mat.mat',...
%'twoChannel2_LOMO_10lda1004020102010100500.0310000mode1_norm3outvectors_phrase_win5_threshold0_size100autoEncodeSentences_trainLevel3_pairs_hiddensizes10040_maxepochs2010100_trainsplit2000_precise0.mat.mat'

};

imagesRun={
%'XQDA-autoEncode2d-0-4-2000-pairs-3-1500-200-100-50-100-0-50-50-mirroredpca.mat',...
%'XQDA-autoEncode2d-0-4-2000-pairs-3-1500-200-100-50-100-0-50-50-nonepca.mat',...
%'XQDA-autoEncode2d-0-4-2000-pairs-3-1500-200-100-50-100-0-50-50-mirroredlda.mat',...
%'XQDA-autoEncode2d-0-4-2000-pairs-3-1500-200-100-50-100-0-50-50-nonelda.mat',...
%'XQDA-autoEncode2d-0-4-2000-pairs-3-1500-200-100-50-100-0-50-50-none.mat',...
%'XQDA-autoEncode2d-0-4-2000-pairs-3-1500-200-100-50-100-0-50-50-mirrored.mat'
%'XQDA-LOMO-0-4-2000-pairs-3-1500-200-100-50-100-0-50-50-mirroredpca.mat',...
%'XQDA-LOMO-0-4-2000-pairs-3-1500-200-100-50-100-0-50-50-nonepca.mat',...
%'XQDA-LOMO-0-4-2000-pairs-3-1500-200-100-50-100-0-50-50-mirroredlda.mat',...
%'XQDA-LOMO-0-4-2000-pairs-3-1500-200-100-50-100-0-50-50-nonelda.mat',...
%'XQDA-LOMO-0-4-2000-pairs-3-1500-200-100-50-100-0-50-50-none.mat',...
%'XQDA-LOMO-0-4-2000-pairs-3-1500-200-100-50-100-0-50-50-mirrored.mat'

'XQDA_ALEX_3_60_0.txt',...
'XQDA_VGG_3_60_0.txt',...
'XQDA_LOMO_3_60_0.txt'
};

presetLabels={
   
    'ALEXNET. 60% Training Samples. ResizeMethod READALL',...
     'VGGNET. 60% Training Samples. ResizeMethod READALL',...
     'LOMO. 60% Training Samples. ResizeMethod READALL'

   %'AutoEncodeImg. Extended using Mirrored. Reduced using PCA',...
   % 'AutoEncodeImg. Not Extended. Reduced using PCA',...
   % 'AutoEncodeImg. Extended using Mirrored. Reduced using LDA',...
   % 'AutoEncodeImg. Not Extended. Reduced using LDA',...
   % 'AutoEncodeImg. Extended using Mirrored. Not Reduced',...
   % 'AutoEncodeImg. Not Extended. Not Reduced'
    %'LOMO. Extended using Mirrored. Reduced using PCA',...
    %'LOMO. Not Extended. Reduced using PCA',...
    %'LOMO. Extended using Mirrored. Reduced using LDA',...
    %'LOMO. Not Extended. Reduced using LDA',...
    %'LOMO. Not Extended. Not Reduced',...
    %'LOMO. Extended using Mirrored. Not Reduced'
};


%{
presetLabels={
    'FalsePositiveRatio 4 Learning Rate 0.01',...
    'FalsePositiveRatio 4 Learning Rate 0.02',...
    'FalsePositiveRatio 4 Learning Rate 0.03',...
    'FalsePositiveRatio 10 Learning Rate 0.01',...
    'FalsePositiveRatio 10 Learning Rate 0.02',...
    'FalsePositiveRatio 10 Learning Rate 0.03',...    
};
%}

%{
presetLabels={
    'FalsePositiveRatio 1',...
    'FalsePositiveRatio 2',...
    'FalsePositiveRatio 3',...
    'FalsePositiveRatio 4',...
    'FalsePositiveRatio 5',...
    'FalsePositiveRatio 10'
};
%}
%{
presetLabels={
    'LOMO. Precise. Extended using mirrored. Reduced using LDA',...
    'LOMO. Precise. Extended using mirrored. Reduced using PCA',...
    'LOMO. Precise. Not Extended. Reduced using LDA',...
    'LOMO. Precise. Not Extended. Reduced using PCA',...
    'LOMO. Non-Precise. Extended using mirrored. Reduced using LDA',...
    'LOMO. Non-Precise. Extended using mirrored. Reduced using PCA',...
    'LOMO. Non-Precise. Not Extended. Reduced using LDA',...
    'LOMO. Non-Precise. Not Extended. Reduced using PCA'
};
%}
%{
presetLabels={
    'AutoEncodeImg. Precise. Extended using mirrored. Reduced using LDA',...
    'AutoEncodeImg. Precise. Extended using mirrored. Reduced using PCA',...
    'AutoEncodeImg. Precise. Not Extended. Reduced using LDA',...
    'AutoEncodeImg. Precise. Not Extended. Reduced using PCA',...
    'AutoEncodeImg. Non-Precise. Extended using mirrored. Reduced using LDA',...
    'AutoEncodeImg. Non-Precise. Extended using mirrored. Reduced using PCA',...
    'AutoEncodeImg. Non-Precise. Not Extended. Reduced using LDA',...
    'AutoEncodeImg. Non-Precise. Not Extended. Reduced using PCA'
};

%}


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
                                    switch(s)
                                        case 1
                                            color='';
                                            widthy=1.5;
                                        case 2
                                            color=':';
                                            widthy=1.5;
                                        case 3
                                            color='';
                                            widthy=1.5;
                                    end
                                    plot(1 : size(meanCms,2), meanCms,color,'LineWidth',widthy)
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
        config{1}
        title(sprintf('CMS Curve for CUHK03 Image Matching'))
        %title(sprintf('CMS Curve for CUHK03 Image Matching with %s', config{1}))
        xlabel('No. Ranks of ordered Gallery Images') % x-axis label
        ylabel('% Probe Images that contain match within that rank') % y-axis label
        labels=labels(~cellfun('isempty',labels))  
        legend(labels);
        

    case SENTENCES
        %% Create given configuration for all graphs shown
        config=strcat('feature', sentenceFeature,'trainlevel', sentenceTrainLevel, 'sentencearrange',sentenceArrange, 'precision', sentencePrecise);
        vals=[sentenceClassifier{1},sentenceSizes(1),sentenceNormalisation(1), sentenceModes(1), sentenceWindows(1), sentenceThresholds(1), [sentenceHiddenSizes(1)],[sentenceMaxEpochs(1)],sentenceTrainSplit(1)];
        descrips={'classifier ','size ','normalisation method ', 'mode ','window ', 'thresholds','hiddensizes ', 'max epochs', 'trainingsplit'};
        dimensions=[length(sentenceClassifier),length(sentenceSizes),length(sentenceNormalisation), length(sentenceModes), length(sentenceWindows), length(sentenceThresholds),size(sentenceHiddenSizes,1),size(sentenceMaxEpochs,1),length(sentenceTrainSplit)];
        idx=find(dimensions==1);
        varyIndexes=setdiff([1:size(dimensions,2)],idx);
        %Config are things that are constant throughout experiment
        for i=1:length(idx)
           %descrips(idx(i))
           config=strcat(config,descrips(idx(i)),num2str(vals(idx(i))));
        end
        labels=cell(length(sentenceClassifier)*length(sentenceSizes)*length(sentenceNormalisation)*length(sentenceModes)*length(sentenceWindows)*length(sentenceThresholds)*size(sentenceHiddenSizes,1)*size(sentenceMaxEpochs,1)*length(sentenceTrainSplit),1);
        %% Find if images have been extracted and show results
        for c=1:length(sentenceClassifier)
           for s= 1:length(sentenceSizes)
               for n=1:length(sentenceNormalisation)
                  for m=1:length(sentenceModes)
                      for w=1:length(sentenceWindows)
                          for t=1:length(sentenceThresholds)
                              for h=1:size(sentenceHiddenSizes,1)
                                  for e=1:size(sentenceMaxEpochs,1)
                                      for ts=1:length(sentenceTrainSplit)
                                          %Get variedparts
                                          name='';
                                                                                                                                            
                                          for v= 1: length(varyIndexes)
                                              switch (varyIndexes(v))
                                                  case 1
                                                      name=strcat(name,'classifier-',sentenceClassifier(c));
                                                  case 2
                                                      name=strcat(name,'size-',num2str(sentenceSizes(s)));
                                                  case 3
                                                      name=strcat(name,'normalisation-',num2str(sentenceNormalisation(n)));
                                                  case 4
                                                      name=strcat(name,'mode-',num2str(sentenceModes(m)));
                                                  case 5
                                                      name=strcat(name,'windows-',num2str(sentenceWindows(w)));
                                                  case 6
                                                      name=strcat(name,'thresholds-',num2str(sentenceThresholds(t)));
                                                  case 7
                                                      name=strcat(name,'hiddenSizes-',num2str(sentenceHiddenSizes(h,1)),num2str(sentenceHiddenSizes(h,2)));
                                                  case 8
                                                      name=strcat(name,'maxepochs-',num2str(sentenceMaxEpochs(e,1)),num2str(sentenceMaxEpochs(e,2)),num2str(sentenceMaxEpochs(e,3)));
                                                  case 9
                                                      name=strcat(name,'trainsplit-',num2str(sentenceTrainSplit(ts)));
                                              end
                                          end
                                          fileNames={sentenceClassifier{c},char(strcat('mode',num2str(sentenceModes(m)))),char(strcat('norm',num2str(sentenceNormalisation(n)),'outvectors')),'phrase',char(strcat('win',num2str(sentenceWindows(w)))),char(strcat('threshold',num2str(sentenceThresholds(t)))),char( strcat('size',num2str(sentenceSizes(s)),'autoEncodeSentences')),char(strcat('trainLevel',sentenceTrainLevel)), sentenceArrange, char(strcat('hiddensizes',num2str(sentenceHiddenSizes(h,1)),num2str(sentenceHiddenSizes(h,2)))), strcat('maxepochs', num2str(sentenceMaxEpochs(e,1)), num2str(sentenceMaxEpochs(e,2)), num2str(sentenceMaxEpochs(e,3))),char(strcat('trainsplit', num2str(sentenceTrainSplit(ts)))),char(strcat('precise', num2str(sentencePrecise))) };
                                          fileName=strjoin(fileNames,'_');
                                          fileName=strcat('../results/sentences/',fileName,'.mat');
                                            if (exist(fileName, 'file') == 2) 
                                                fprintf('filename %s found, extracting now\n',fileName);
                                                load( fileName);
                                                if(meanCms(1,200)>=sentenceTop || sentenceTop==0)
                                                    labels{ts+ length(sentenceTrainSplit)*(ts-1)+length(sentenceTrainSplit)*size(sentenceMaxEpochs,1)*(e-1)+length(sentenceTrainSplit)*size(sentenceMaxEpochs,1)*size(sentenceHiddenSizes,1)*(h-1)+length(sentenceTrainSplit)*size(sentenceMaxEpochs,1)*size(sentenceHiddenSizes,1)*length(sentenceThresholds)*(t-1)+length(sentenceTrainSplit)*size(sentenceMaxEpochs,1)*size(sentenceHiddenSizes,1)*length(sentenceThresholds)*length(sentenceWindows)*(w-1)+length(sentenceTrainSplit)*size(sentenceMaxEpochs,1)*size(sentenceHiddenSizes,1)*length(sentenceThresholds)*length(sentenceWindows)*length(sentenceModes)*(m-1)+length(sentenceTrainSplit)*size(sentenceMaxEpochs,1)*size(sentenceHiddenSizes,1)*length(sentenceThresholds)*length(sentenceWindows)*length(sentenceModes)*length(sentenceNormalisation)*(n-1)+length(sentenceTrainSplit)*size(sentenceMaxEpochs,1)*size(sentenceHiddenSizes,1)*length(sentenceThresholds)*length(sentenceWindows)*length(sentenceModes)*length(sentenceNormalisation)*length(sentenceSizes)*(s-1)+length(sentenceTrainSplit)*size(sentenceMaxEpochs,1)*size(sentenceHiddenSizes,1)*length(sentenceThresholds)*length(sentenceWindows)*length(sentenceModes)*length(sentenceNormalisation)*length(sentenceSizes)*length(sentenceClassifier)*(c-1),1}=name; 
                                                    if(sentenceTop~=0)
                                                        sentenceTop=meanCms(1,200);
                                                    end
                                                    plot(1 : size(meanCms,2), meanCms,'LineWidth',1.5)
                                                    hold on;
                                                end
                                            else
                                                fprintf('filename %s not found\n',fileName);
                                            end
                                      end
                                  end
                              end
                          end
                      end
                  end
               end
           end
        end
        labels
        title(sprintf('CMS Curve for Sentence Matching'))
        %title(sprintf('CMS Curve for Sentence Matching with %s', config{1}))
        xlabel('No. Ranks of ordered Gallery Sentences') % x-axis label
        ylabel('% Probe Sentences that contain match within that rank') % y-axis label
        labels=labels(~cellfun('isempty',labels))  
        legend(labels);
    
    case SENTENCESPECIFIC
        labels=cell(length(sentencesRun),1);
        
        for i=1:length(sentencesRun)
            props(i,:)=strsplit(sentencesRun{i},'_');
        end
        for e=1:size(props,2)
            for i=1:length(sentencesRun)
               repeatdetected=0;
               for u=1:length(sentencesRun)
                    if( strcmp(props{i,e},props{u,e}))
                        repeatdetected=repeatdetected+1;
                    end
               end
               if(repeatdetected==(length(sentencesRun)))
                  %props{i,e}=''; 
                  repeatVal(i,e)=1;
               end
            end
        end
        for e=1:size(props,2)
            for i=1:length(sentencesRun)
                if(repeatVal(i,e))
                    props{i,e}=' '; 
                end
            end
        end
        
        for i=1:length(sentencesRun)
            fileName=strcat('../results/sentences/',sentencesRun{i},'.mat');
            if (exist(fileName, 'file') == 2) 
                fprintf('filename %s found, extracting now\n',fileName);
                labels{i}=strcat(props{i,:});
                load( fileName);
                plot(1 : size(meanCms,2), meanCms,'LineWidth',1.5)
                hold on;
            else
            	fprintf('filename %s not found\n',fileName);
            end  
        end
        
        title(sprintf('CMS Curve for Sentence Matching'))
        %title(sprintf('CMS Curve for Sentence Matching with %s', config{1}))
        xlabel('No. Ranks of ordered Gallery Sentences') % x-axis label
        ylabel('% Probe Sentences that contain match within that rank') % y-axis label
        labels=labels(~cellfun('isempty',labels))  
        legend(labels);
        
    case SENTENCEIMAGES
        labels=cell(length(sentenceImagesRun),1);
        
        for i=1:length(sentenceImagesRun)
            props(i,:)=strsplit(sentenceImagesRun{i},'_');
        end
        for e=1:size(props,2)
            for i=1:length(sentenceImagesRun)
               repeatdetected=0;
               for u=1:length(sentenceImagesRun)
                    if( strcmp(props{i,e},props{u,e}))
                        repeatdetected=repeatdetected+1;
                    end
               end
               if(repeatdetected==(length(sentenceImagesRun)))
                  %props{i,e}=''; 
                  repeatVal(i,e)=1;
               end
            end
        end
        for e=1:size(props,2)
            for i=1:length(sentenceImagesRun)
                if(repeatVal(i,e))
                    props{i,e}=' '; 
                end
            end
        end
        
        for i=1:length(sentenceImagesRun)
            fileName=strcat('../results/sentenceImages/',sentenceImagesRun{i});
            if (exist(fileName, 'file') == 2) 
                fprintf('filename %s found, extracting now\n',fileName);
                labels{i}=strcat(props{i,:});
                load( fileName);
                plot(1 : size(meanCms,2), meanCms,'LineWidth',1.5)
                hold on;
            else
            	fprintf('filename %s not found\n',fileName);
            end  
        end
        plot(1 : 100, 0.01:0.01:1,'k--','LineWidth',1.5)
        hold on;
        title(sprintf('CMS Curve for Sentence Image Matching'))
        %title(sprintf('CMS Curve for Sentence Matching with %s', config{1}))
        xlabel('No. Ranks of ordered Gallery Sentences') % x-axis label
        ylabel('% Probe Images that contain match within that rank') % y-axis label
        labels=labels(~cellfun('isempty',labels)) 
        
        labels=presetLabels;
        legend(labels);
        
     case IMAGESPECIFIC

        for i=1:length(imagesRun)
            fileName=strcat('../results/images/',imagesRun{i});
            if (exist(fileName, 'file') == 2) 
                fprintf('filename %s found, extracting now\n',fileName);
                if(isequal(strfind(fileName,'60'),[]))
                    load( fileName);
                else
                    meanCms=table2array(readtable(fileName));
                end
                plot(1 : size(meanCms,2), meanCms,'LineWidth',1.5)
                hold on;
            else
            	fprintf('filename %s not found\n',fileName);
            end  
        end
        plot(1 : 679, 1/679:1/679:1,'k--','LineWidth',1.5)
        hold on;       
        title(sprintf('CMS Curve for Image Matching'))
        %title(sprintf('CMS Curve for Sentence Matching with %s', config{1}))
        xlabel('No. Ranks of ordered Gallery Images') % x-axis label
        ylabel('% Probe Images that contain match within that rank') % y-axis label
        labels=presetLabels;
        legend(labels);   
    
end


    
 