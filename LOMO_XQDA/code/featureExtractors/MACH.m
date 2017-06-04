function descriptors = MACH(imageStore, options)
%% function Descriptors = MACH(images, options)
% Function for the machine learning feature extraction
%
% Input:
%   <images>: a set of n RGB color images. Size: [h, w, 3, n]

% Output:
%   descriptors: the extracted LOMO descriptors. Size: [d, n]
% 
% Example:
%     I = imread('../images/000_45_a.bmp');
%     descriptor = LOMO(I);

%% set parameters, check system
if nargin >= 2
    if isfield(options,'trainSplit') && ~isempty(options.trainSplit) && isscalar(options.trainSplit) && isnumeric(options.trainSplit) && options.trainSplit > 0
        trainSplit = options.trainSplit;
        fprintf('Training percentage of images is %d.\n', trainSplit);
    end
end
t0 = tic;
% Get GPU device information
%deviceInfo = gpuDevice;

% Check the GPU compute capability
%computeCapability = str2double(deviceInfo.ComputeCapability);
%assert(computeCapability >= 3.0, ...
 %   'This example requires a GPU device with compute capability 3.0 or higher.')

%% create image datastores 
[imagesTrain,imagesTest]= splitEachLabel(imageStore,trainSplit);


%% extract Features: descriptors
net = alexnet;


%% finishing, clear temp vars, create descriptors
descriptors='hi'

feaTime = toc(t0);
meanTime = feaTime / size(images, 4);
fprintf('MACH feature extraction finished. Running time: %.3f seconds in total, %.3f seconds per image.\n', feaTime, meanTime);
end

%{
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
iMAGE cATEGORY CLASSIFICATION
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Get GPU device information
%deviceInfo = gpuDevice;

% Check the GPU compute capability
%computeCapability = str2double(deviceInfo.ComputeCapability);
%assert(computeCapability >= 3.0, ...
 %   'This example requires a GPU device with compute capability 3.0 or higher.')

url = 'http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz';
% Store the output in a temporary folder
outputFolder = fullfile(tempdir, 'caltech101'); % define output folder
if ~exist(outputFolder, 'dir') % download only once
    disp('Downloading 126MB Caltech101 data set...');
    untar(url, outputFolder);
end
rootFolder = fullfile(outputFolder, '101_ObjectCategories');
categories = {'airplanes', 'ferry', 'laptop'};
imds = imageDatastore(fullfile(rootFolder, categories), 'LabelSource', 'foldernames');
%imageDatastore labels is empty if not specified
tbl = countEachLabel(imds)
%%%%%%%%%%%%Balance out no. image labels for training
%Equal positive and negative
minSetCount = min(tbl{:,2}); % determine the smallest amount of images in a category

% Use splitEachLabel method to trim the set.
imds = splitEachLabel(imds, minSetCount, 'randomize');
% Find the first instance of an image for each category
airplanes = find(imds.Labels == 'airplanes', 1);

% Notice that each set now has exactly the same number of images.
countEachLabel(imds)

%}



