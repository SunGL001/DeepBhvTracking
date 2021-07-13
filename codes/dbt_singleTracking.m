% This is an intergrated  approach for tracking the position of lab animals for single file. 
% Import the video path, load the corresponding detector, and define the tracking area manually. 
% Finaly, the bounding-box of animal can be tracked by YOLO and the center of animal can be calculated by background substration in the bounding-box automatically. 
% Caution: background substraction is highly correlated with color of target with background (lines 11 and 93-97).
% Developed by GuangLong Sun et al., 2021.
% Please cite: DeepBhvTracking: A Novel Behavior Tracking Method for Laboratory Animals Based on Deep Learning
clear; close all; clc
%% set neccessary parameters (this part needs to be set manually before tracking)
fn='D:\DeepBhvTracking\demo_video.mp4'; % video path
fd='D:\DeepBhvTracking\demo_detecor_BlackMice.mat'; % detector path
animalColor = 2;  % '1' means black animal removing background and '2' means white animal. details are shown in lines 93-97.
% If the video has been converted to mat format, please next load the name of video variable. If not, ignore it.
loadvar = 'VideoVariable'; % name of video variable in matfile
%% load data
[fPath,fName,ext]=fileparts(fn);
tic
[bhvdata,~]=dbt_bhvread(fn,loadvar);
toc
%% deep learning need rgb data format, if demension of data is not 4 then change it to 4 
dim=size(bhvdata);
if length(dim)<4
    bhvdata=repmat(bhvdata,[1 1 1 3]);
    bhvdata=permute(bhvdata,[1 2 4 3]);
end
dim=size(bhvdata);
%% define mask as tracking area
% Caution: close the window after defining mask 
figure
imshow(bhvdata(:,:,:,1))
Y= drawpolygon;
mazeMask = Y.createMask;
delete(Y)
close 
%% if some items are simlilar to target, then reomove them from mask
% figure
% imshow(bhvdata(:,:,:,1),[])
% mazeMask2=[];
% for i=1:2     % the number of simlilar items
% Y2=drawrectangle;
% mazeMask2{i} = Y2.createMask;
% end
% delete(Y2)
% close
% mazeMask2=logical(sum(cat(3,mazeMask2{:}),3));
% mazeMask=(mazeMask & ~mazeMask2);
%% remove background and unrealted areas (set to white)
mazeMask=mazeMask(:);
bhvdata=reshape(bhvdata,[dim(1)*dim(2),dim(3:4)]);
bhvdata(~mazeMask,:,:)=255;
bhvdata=reshape(bhvdata,dim);
%% load detector
load(fd) % load detector
%% get bounding-boxes
threshold =0.2; % threshold of bounding-box
disp('detcection by deep learning:..........Start')
[bboxes,scores] = arrayfun(@(i) detect(detector,bhvdata(:,:,:,i),'Threshold',threshold),1:size(bhvdata,4),'UniformOutput',false);
toc
disp('detcection by deep learning:..........Done')
%% updata bounding-box (bounding-box with the largest p-value is selected)
bboxes=cellfun(@(x,y) [x(y==max(y),:) max(y)],bboxes, scores, 'UniformOutput', false);
for i=1:length(bboxes)
    temp=bboxes{i};
    if isempty(temp)
        bboxes{i}=zeros(1,5);
    end
end
bboxes=double(cat(1,bboxes{:}));
%% view accurity using yolo only
% bhvdata2=arrayfun(@(i)  insertObjectAnnotation(bhvdata(:,:,:,i),'rectangle',bboxes(i,1:4),bboxes(i,5),'Color','red'),...
%     1:size(bhvdata,4),'UniformOutput',false);
% bhvdata2=cat(4,bhvdata2{:});
% implay(bhvdata2)
%% calculate the center of animal using background substraction; 
disp('background method to redefine center>>>>>>>>>>>>>>>>>>>>>>>>>>>>>start')
% the dimension for this method should be 3, so change it to 3
if length(size(bhvdata)) > 3
    bhvdata=arrayfun(@(i) {rgb2gray(bhvdata(:,:,:,i))},1:size(bhvdata,4));
    bhvdata=cat(3,bhvdata{:});
end
dim=size(bhvdata);
bhvdata=reshape(bhvdata,[],size(bhvdata,3));
%% create mask
scale = 1.5; % enlarge the bounding box for  cover the animal completely
aR =bboxes(:,1:4); aR (:,1:2)=aR(:,1:2)-aR(:,3:4)*(scale-1); 
aR(:,3:4)=aR(:,1:2)+aR(:,3:4)*scale;
aR1=aR(:,[1:2 3 2 3 4 1 4 1:2]);
AllBW=arrayfun(@(b)  poly2mask(aR1(b,1:2:end),aR1(b,2:2:end),dim(1),dim(2)),1:size(aR1,1),'UniformOutput',false);
AllBW=cat(3,AllBW{:}); AllBW=reshape(AllBW,[],dim(3));
% AllBW=reshape(AllBW,dim); implay(AllBW)
%% background
bgImgM = uint8(mean(bhvdata,2));
%% image remove background
if animalColor==1
    bhvdata_bg=bgImgM-bhvdata; % for dark animal in light background
elseif animalColor==2
    bhvdata_bg=bhvdata-bgImgM; % for white animal in dark background
end
bhvdata_bg1=reshape(bhvdata_bg,dim);
%implay(bhvdata_bg1)
clear bhvdata
%% rectangel imaging
disp('backgroud remove method retracking..............')
tic
rec_image=zeros(dim); rec_image=reshape(rec_image,[],dim(3));
rec_image(AllBW)=bhvdata_bg(AllBW);
clear AllBW
%% find the binary position of animal
thrP = 85;  % threshold of removing background
mouseWidth = 2; 
mouseSE = strel('disk',round(mouseWidth/2));
bw=arrayfun(@(i)...
    rec_image(:,i)>multithresh(rec_image(rec_image(:,i)>prctile(rec_image(:,i),thrP),i)),...
    1:dim(3),'UniformOutput',false);
bw=cellfun(@(x) imreconstruct(imerode(x,mouseSE),x),bw,'UniformOutput',false);
bw=cat(2,bw{:}); bw=reshape(bw,dim);
toc
clear rec_image
% implay(bw)  % binary position of animal
idx=find(bboxes(:,5)==0);
disp(['lost frames.......',num2str(length(idx))])
%% calculate centroids
[py,px,pt] = ind2sub(dim,find(bw));
mCenA = NaN(dim(3),2);
mCenA(:,1) = accumarray(pt,px,[],@mean,NaN);
mCenA(:,2) = accumarray(pt,py,[],@mean,NaN);
%% save
movData.mCenA=mCenA;
movData.bw=bw;
movData.bboxes=bboxes;
movData.mazeMask=mazeMask;
movData.bgImgM=bgImgM;
movData.dim=dim;
if strcmp(ext,'.mat')
    save([fn(1:end-3),'mat'],'movData','-append')
else
    save([fn(1:end-3),'mat'],'movData','-v7.3')
end
%% plot
bgImgM=reshape(bgImgM,dim(1),dim(2));
figure;
imshow(bgImgM)
hold on
plot(mCenA(:,1),mCenA(:,2))
toc
