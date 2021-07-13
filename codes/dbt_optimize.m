% This is an approach for saving and labeling undetected and incorrect frames to optimize detector.
% Cautious: Image Label App base on the computer vision toolbox is used and
% the details are available online at https://www.mathworks.com/help/vision/ref/imagelabeler-app.html.
% Label the images by a rectangular box and export labels to workspace after finishing labelling.
% Developed by GuangLong Sun et al., 2021.
% Please cite: DeepBhvTracking: A Novel Behavior Tracking Method for Laboratory Animals Based on Deep Learning
clear; close all; clc 
%% get file
fn='D:\DeepBhvTracking\demo_video.mp4'; % video path
[fPath,fName,ext]=fileparts(fn);
fmat=fullfile(fPath,[fName,'.mat']);
%% get data
mat=matfile(fmat);  obj=VideoReader(fn);
load (fmat,'movData')
bboxes=movData.bboxes;  mCenA=movData.mCenA;
%% save undectect images
idx=find(bboxes(:,5)==0);
undectedVideo=cell(length(idx),1);
parfor i=1:length(idx)
    try
        frame=mat.VideoVariable(:,:,:,idx(i)); % name of video variable in matfile
    catch     
        frame=read(obj,idx(i));
    end
    undectedVideo{i,1}=frame;
end
undectedVideo=cat(4,undectedVideo{:}); % create undectedVideo
%implay(undectedVideo)
savefolder='C:\Users\dell\Desktop\videos\images\FigS';  % output file path
pnum = 50; step= ceil(length(idx)/pnum); % the number of output picture
j=0;
for i=1:step:length(idx)  % save all images by default
    j=j+1;
    temp=undectedVideo(:,:,:,i);
    savefile=fullfile(savefolder,['S1_',num2str(j,'%04d'),'.jpg']);
    imwrite(temp,savefile)
end
%% save uncorrect images
idx=find(diff(mCenA(:,1))>50);
uncorrectVideo=cell(length(idx),1);
parfor i=1:length(idx)
   try
        frame=mat.VideoVariable(:,:,:,idx(i)); % name of video variable in matfile
    catch     
        frame=read(obj,idx(i));
    end
    uncorrectVideo{i,1}=frame;
end
uncorrectVideo=cat(4,uncorrectVideo{:}); % create uncorrectVideo
% implay(uncorrectVideo)
savefolder='C:\Users\dell\Desktop\M_S1';  % output file path
pnum = 200; step=ceil(length(idx)/pnum); % the number of output picture
j=0;
for i=1:step:length(idx)  % save all images by default
    j=j+1;
    temp=uncorrectVideo(:,:,:,i);
    savefile=fullfile(savefolder,['S2_',num2str(j,'%04d'),'.jpg']);
    imwrite(temp,savefile)
end
%% label data manually
% Cautious: export labels to workspace after finishing labelling
imageLabeler(savefolder)
%% save
mouseDataset=[gTruth.DataSource.Source gTruth.LabelData];
mouseDataset.Properties.VariableNames{1} = 'imageFilename';
mouseDataset.Properties.VariableNames{2} = 'mouse';
savename='mouseDatasetS1.mat';
save(fullfile(savefolder,savename),'mouseDataset')
%% merge dataset 
fn='C:\Users\dell\Desktop\videos\images\mouseDataset.mat'; % previous dataset path
mouseDataset1=load(fn);
mouseDataset1=mouseDataset1.mouseDataset;
mouseDataset=[mouseDataset1;mouseDataset];
[fpath,fname,ext] = fileparts(fn);
save(fullfile(fpath,'mouseDataset_merge.mat'),'mouseDataset')


