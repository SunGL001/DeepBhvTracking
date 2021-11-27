% This is an intergrated  approach for create labeled video after tracking.
% Developed by GuangLong Sun et al., 2021.
% Please cite: DeepBhvTracking: A Novel Behavior Tracking Method for Laboratory Animals Based on Deep Learning
clear; close all; clc 
%% get file
fn='D:\YOLO\forPublish_20210713\demo_video.mp4'; % video path
[fPath,fName,ext]=fileparts(fn);
fmat=fullfile(fPath,[fName,'.mat']) ;
mat=matfile(fmat);  
loadvar = 'VideoVariable'; % name of video variable in matfile
%% get data
[bhvdata,~]=dbt_bhvread(fn,loadvar);
load (fmat,'movData')
mCenA=movData.mCenA;
%% create labeled video
bhvdata2=arrayfun(@(i) {insertShape(bhvdata(:,:,:,i),'circle',[mCenA(i,1),mCenA(i,2),1],...
    'Color','red','LineWidth',3)},1:size(bhvdata,4));
bhvdata2=cat(4,bhvdata2{:});
% implay(bhvdata2)
%% save
savefig=fullfile(fPath,[fName,'_Labeld']);
writerObj=VideoWriter(savefig);
writerObj.FrameRate=30; 
open(writerObj);
writeVideo(writerObj,bhvdata2);
close(writerObj);

 