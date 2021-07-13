% This is an intergrated  approach for manually tracking the position of
% animal of undetected and uncorrect frames by deep learning.
% Cautious: run the section 'tracking uncorrect frames'repeatedly until there are no uncorrect frames
% Developed by GuangLong Sun et al., 2021.
% Please cite: DeepBhvTracking: A Novel Behavior Tracking Method for Laboratory Animals Based on Deep Learning
clear; close all; clc 
%% get file
fn='D:\DeepBhvTracking\demo_video.mp4'; % video path
[fPath,fName,ext]=fileparts(fn);
fmat=fullfile(fPath,[fName,'.mat']) ;
mat=matfile(fmat);  
try
obj=VideoReader(fn);
end 
%% get data
 load (fmat,'movData')
 bboxes=movData.bboxes;
 mCenA=movData.mCenA;
 bgImgM=movData.bgImgM;
 dim=movData.dim;
 %% tracking undected frames
idx=find(bboxes(:,5)==0);
% idx=find(isnan(mCenA(:,1)));
if ~isempty(idx)
    tcr=[];
    figure('Position',[200 200 1000 800 ])
    for j=1:length(idx)
        disp([num2str(idx(j)),'.....',num2str(j),'....',num2str(length(idx))])
        hold on
        try
            imshow(mat.bhv_video1(:,:,:,idx(j))) % name of video variable in matfile   VideoVariable
        catch
            imshow(read(obj,idx(j)))
        end
        tc1(idx(j),:)=ginput(1);
        clear cla
    end
    tcr(idx,:)=tc1(idx,:);
    mCenA(idx,:)=tcr(idx,:);
    close all
end
%%  tracking uncorrect frames
% Cautious: run repeatedly until there are no uncorrect frames
idx1=[];
idx=find(diff(mCenA(:,1))>50 | diff(mCenA(:,2))>50); % the gap between two adjacent frames is larger than 50 is selected. 
if ~isempty(idx)
idx=[idx-1 idx idx+1]; idx=sort(idx(:)); idx=setdiff(idx,idx1);
if ~isempty(idx)
    tcr=[];
    figure('Position',[200 200 1000 800 ])
    for j=1:length(idx)
        disp([num2str(idx(j)),'.....',num2str(j),'....',num2str(length(idx))])
        hold on
        try
            imshow(mat.bhv_video1(:,:,:,idx(j))) % name of video variable in matfile
        catch
            imshow(read(obj,idx(j)))
        end
        tc1(idx(j),:)=ginput(1);
        clear cla
    end
    tcr(idx,:)=tc1(idx,:);
    mCenA(idx,:)=tcr(idx,:);
end
idx1=[idx1;idx];
close all
end
%% figure plot
bgImgM=reshape(bgImgM,dim(1),dim(2));
figure;
imshow(bgImgM)
hold on
plot(mCenA(:,1),mCenA(:,2))
%% save
movData.mCenA=mCenA;
save(fmat,'movData','-append')
