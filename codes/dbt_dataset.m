% This is an approach for getting datasets for training. 
% Cautious: Image Label App base on the computer vision toolbox is used and
% the details are available online at https://www.mathworks.com/help/vision/ref/imagelabeler-app.html.
% Label the images by a rectangular box and export labels to workspace after finishing labelling.
% Developed by GuangLong Sun et al., 2021.
% Please cite: DeepBhvTracking: A Novel Behavior Tracking Method for Laboratory Animals Based on Deep Learning.
clear; close all; clc 
%% extract images from video randomly 
fn = 'D:\DeepBhvTracking\demo_video.mp4';  % video path  
savefolder='D:\DeepBhvTracking';  % output file path of extract frames
obj = VideoReader(fn);
num = 300; % the number of extrat images
step = floor(obj.NumFrames/num);
i = 0;
for j = 1 :step: obj.NumFrames
    i = i + 1;
    frame = read(obj,j);                                 
    imwrite(frame,fullfile(savefolder,[num2str(i,'%04d'),'.jpg']),'jpg'); 
end
%% label data manually
% Cautious: export labels to workspace after finishing labelling
imageLabeler(savefolder)
%% save
mouseDataset=[gTruth.DataSource.Source gTruth.LabelData];
mouseDataset.Properties.VariableNames{1} = 'imageFilename';
mouseDataset.Properties.VariableNames{2} = 'mouse';
save(fullfile(savefolder,'mouseDataset.mat'),'mouseDataset')




