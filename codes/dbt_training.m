% This is an intergrated approach for training detector using YOLO. 
% Cautious: Download pre-trained network in advance (resnet50,resnet18, mobilenetv2 or others) 
% and details are shown in https://www.mathworks.com/help/deeplearning/ug/pretrained-convolutional-neural-networks.html.
% Developed by GuangLong Sun et al., 2021.
% Please cite: DeepBhvTracking: A Novel Behavior Tracking Method for Laboratory Animals Based on Deep Learning
close all; clc;clear
%% load labeled images
fn='D:\DeepBhvTracking\mouseDataset.mat';    % dataset path
[fPath,fName,ext]=fileparts(fn);
load(fn);
%% change images path in dataset if moving the image folder after labeling
% PathName = 'C:\Users\dell\Desktop\newfolder';
% mat=matfile(fn);
% temp=mat.mouseDataset;
% imageFilename=cell(size(temp,1),1);
% mouse=cell(size(temp,1),1);
% box=table2cell(temp(:,2));
% for i=1:size(temp,1)
% [iPath,iName,iext]=fileparts(cell2mat(temp{i,1}));
% imageFilename{i}=fullfile(PathName,[iName,iext]);
% mouse{i}=box{i};
% end
% mouseDataset=table(imageFilename,mouse); 
% save(fullfile(PathName,'mouseDataset.mat'),'mouseDataset')
%% devide dataset for training, validation and test
rng(0);
shuffledIndices = randperm(size(mouseDataset,1));
idx = floor(0.7 * length(shuffledIndices) ); %training data
trainingIdx = 1:idx;
trainingDataTbl = mouseDataset(shuffledIndices(trainingIdx),:);
validationIdx = idx+1 : idx + 1 + floor(0.1 * length(shuffledIndices) ); % validation data
validationDataTbl = mouseDataset(shuffledIndices(validationIdx),:);
testIdx = validationIdx(end)+1 : length(shuffledIndices); % tset data
testDataTbl = mouseDataset(shuffledIndices(testIdx),:);
%% create datastore
imdsTrain = imageDatastore(trainingDataTbl{:,'imageFilename'});
bldsTrain = boxLabelDatastore(trainingDataTbl(:,'mouse'));
imdsValidation = imageDatastore(validationDataTbl{:,'imageFilename'});
bldsValidation = boxLabelDatastore(validationDataTbl(:,'mouse'));
imdsTest = imageDatastore(testDataTbl{:,'imageFilename'});
bldsTest = boxLabelDatastore(testDataTbl(:,'mouse'));
% combine images with box labels
trainingData = combine(imdsTrain,bldsTrain);
validationData = combine(imdsValidation,bldsValidation);
testData = combine(imdsTest,bldsTest);
%% view labelled images
% data = read(trainingData);
% I = data{1};
% bbox = data{2};
% annotatedImage = insertShape(I,'Rectangle',bbox);
% annotatedImage = imresize(annotatedImage,2);
% figure
% imshow(annotatedImage)
%% create yolo struction
% Cautious: the first two dimensions of input size should be equal but can be customized to be multiples of 32.
% Larger sizes have higher accuracy but slower speed. The third dimension represents the image is in RGB format.
inputSize = [480 480 3];  % resize input size of images.
numClasses = width(mouseDataset)-1;
trainingDataForEstimation = transform(trainingData,@(data)dbt_preprocessData(data,inputSize));
numAnchors = 7;
[anchorBoxes, meanIoU] = estimateAnchorBoxes(trainingDataForEstimation, numAnchors);
% Cautious: download 'Deep Learning Toolbox Model for ResNet-50 Network' in the mathwork in advance.
featureExtractionNetwork = resnet50; % load pre-trained deep neural network
featureLayer = 'activation_40_relu'; % select featureLayer
% % resnet18
% featureExtractionNetwork = resnet18; 
% featureLayer = 'res5b_branch2a_relu'; 
% % mobilenetv2
% featureExtractionNetwork = mobilenetv2; % 
% featureLayer = 'block_16_expand_relu'; % 
lgraph = yolov2Layers(inputSize,numClasses,anchorBoxes,featureExtractionNetwork,featureLayer); % Connect the YOLO struction
%% data augmentation and pre-process
augmentedTrainingData = transform(trainingData,@dbt_augmentData); % data augmentation
preprocessedTrainingData = transform(augmentedTrainingData,@(data)dbt_preprocessData(data,inputSize)); % resize teh size of images to be the same as input size 
preprocessedValidationData = transform(validationData,@(data)dbt_preprocessData(data,inputSize));
%% view example of augmented and preprocessed data
% augmentedData = cell(4,1);
% for k = 1:4
%     data = read(augmentedTrainingData);
%     augmentedData{k} = insertShape(data{1},'Rectangle',data{2});
%     reset(augmentedTrainingData);
% end
% figure
% montage(augmentedData,'BorderSize',10)

% data = read(preprocessedTrainingData);
% I = data{1};
% bbox = data{2};
% annotatedImage = insertShape(I,'Rectangle',bbox);
% annotatedImage = imresize(annotatedImage,2);
% figure
% imshow(annotatedImage)
%% start training detecor
options = trainingOptions('sgdm', ...
        'MiniBatchSize',8, ....
        'InitialLearnRate',1e-4, ...
        'MaxEpochs',20, ...
        'CheckpointPath',tempdir, ...
        'ValidationData',preprocessedValidationData);
[detector,info] = trainYOLOv2ObjectDetector(preprocessedTrainingData,lgraph,options);
%% test
preprocessedTestData = transform(testData,@(data)dbt_preprocessData(data,inputSize));
detectionResults = detect(detector, preprocessedTestData);
[ap,recall,precision] = evaluateDetectionPrecision(detectionResults, preprocessedTestData);
figure();
plot(recall,precision)
xlabel('Recall')
ylabel('Precision')
grid on
title(sprintf('Average Precision = %.2f',ap))
%% save detector
save(fullfile(fPath,'detector.mat'),'detector')