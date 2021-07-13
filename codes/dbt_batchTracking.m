% This is an intergrated  approach for tracking the position of lab animals for a batch of files. 
% Please be familiar with steps for 'dbt_singleTracking' before using 'dbt_batchTracking'.
% Import the video path, load the corresponding detector, and define the tracking area manually. 
% Finaly, the bounding-box of animal can be tracked by YOLO and the center of animal can be calculated by background substration in the bounding-box automatically. 
% Caution: background substraction is highly correlated with color of target with background (lines 13 and 127-131).
% Developed by GuangLong Sun et al., 2021.
% Please cite: DeepBhvTracking: A Novel Behavior Tracking Method for Laboratory Animals Based on Deep Learning
clear; close all; clc
%% set neccessary parameters (this part needs to be set manually before tracking)
fPath='D:\DeepBhvTracking'; % videos root path
fns=struct2table(dir(fullfile(fPath,'**','*.mp4'))); % Support MOV, MP4, avi, hd5 et,al.
fd='D:\DeepBhvTracking\demo_detecor_BlackMice.mat'; % detector path
animalColor = 1;  % '1' means black animal removing background and '2' means white animal. details are shown in lines 127-131.
% If the video has been converted to mat format, please next load the name of video variable. If not, ignore it. 
loadvar = 'VideoVariable'; % name of video variable in matfile
%% get files
fns=fullfile(fns.folder,fns.name);
nfns=size(fns,1);
%% define maze mask firt
disp('manually get the maze mask.......')
for kk=1:nfns
    disp(['file......',num2str(kk),'......total...',num2str(nfns)])
    tfn=fns{kk};
    try
        mat=matfile(tfn);
        I=mat.VideoVariable(:,:,:,1); % name of video variable in matfile
    catch
        obj=VideoReader(tfn);
        I=read(obj,1);
    end
    figure
    imshow(I)
    Y= drawpolygon;
    mazeMask = Y.createMask;
    delete(Y);
    %% if some items are simlilar to target, then reomove them from mask
    % figure
    % imshow(I)
    % mazeMask2=[];
    % for i=1:2     % the number of simlilar items
    % Y2=drawrectangle;
    % mazeMask2{i} = Y2.createMask;
    % end
    % delete(Y2)
    % close
    % mazeMask2=logical(sum(cat(3,mazeMask2{:}),3));
    % mazeMask=(mazeMask & ~mazeMask2);
    %% save mask
    mazeMask=mazeMask(:);
    movData.mazeMask=mazeMask;
    [filepath,name,ext] = fileparts(tfn);
    if strcmp(ext,'.mat')
        save(tfn,'movData','-append')
    else
        save([fullfile(filepath,name),'.mat'],'movData','-v7.3')
    end
    close 
end
%% batch tracking
disp ('Batch tracking.............')
%% load dectctor and set parameters
load(fd) % load detector
thrP = 85; mouseWidth = 2; % default
mouseSE = strel('disk',round(mouseWidth/2));
%% tracking start
disp ('Batch tracking start:.............')
for kk=1:nfns
    disp(['file......',num2str(kk),'......total...',num2str(nfns)])
    %% get file
    tfn=fns{kk};
    [filepath,name,ext] = fileparts(tfn);
    mtfn=[fullfile(filepath,name),'.mat'];
    mat=matfile(mtfn);
    movData=mat.movData;mazeMask=movData.mazeMask;
    tic
    [bhvdata,~]=dbt_bhvread(tfn,loadvar);
    toc
    %% adjust demension,if demension less than 4, adjust to 4 demension
    dim=size(bhvdata);
            if length(dim)<4
                bhvdata=repmat(bhvdata,[1 1 1 3]);
                bhvdata=permute(bhvdata,[1 2 4 3]);
            end
            %% remove background and unrealted areas (set to white)
            bhvdata=reshape(bhvdata,[dim(1)*dim(2),dim(3:4)]);
            bhvdata(~mazeMask,:,:)=255;
            bhvdata=reshape(bhvdata,dim);
            % implay(bhvdata)
            %% detect position
            disp('detcection by deep learning ..........................')
            tic
            [bboxes,scores] = arrayfun(@(i) detect(detector,bhvdata(:,:,:,i),'Threshold',0.2),1:size(bhvdata,4),'UniformOutput',false);
            toc
            %% correct multi target
            bboxes1=cellfun(@(x,y) [x(y==max(y),:) max(y)],bboxes, scores, 'UniformOutput', false);
            for i=1:length(bboxes1)
                temp=bboxes1{i};
                if isempty(temp)
                    bboxes1{i}=zeros(1,5);
                end
            end
            bboxes=double(cat(1,bboxes1{:}));     
        %% test accurity
        %     bhvdata2=arrayfun(@(i)  insertObjectAnnotation(bhvdata(:,:,:,i),'rectangle',bboxes(i,1:4),bboxes(i,5),'Color','red'),...
        %         1:size(bhvdata,4),'UniformOutput',false);
        %     bhvdata2=cat(4,bhvdata2{:});
        %  implay(bhvdata2)
        %% use background tracking method to redefine the position
        disp('background method to redefine  center')
        if length(size(bhvdata)) > 3
            bhvdata=arrayfun(@(i) {rgb2gray(bhvdata(:,:,:,i))},1:size(bhvdata,4));
            bhvdata=cat(3,bhvdata{:});
        end
        dim=size(bhvdata); [bvr,bvc,bvt]=size(bhvdata);
        bhvdata=reshape(bhvdata,[],size(bhvdata,3));
        %% create mask by deep learning rectangle
        scale = 1.5; % enlarge the bounding box for cover animal completely.
        aR =bboxes(:,1:4); aR (:,1:2)=aR(:,1:2)-aR(:,3:4)*(scale-1);
        aR(:,3:4)=aR(:,1:2)+aR(:,3:4)*scale;
        aR1=aR(:,[1:2 3 2 3 4 1 4 1:2]);
        AllBW=arrayfun(@(b)  poly2mask(aR1(b,1:2:end),aR1(b,2:2:end),dim(1),dim(2)),1:size(aR1,1),'UniformOutput',false);
        AllBW=cat(3,AllBW{:}); AllBW=reshape(AllBW,[],dim(3));
        % AllBW=reshape(AllBW,dim); implay(AllBW)
        %% background image
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
        %% rectangle imaging
        disp('backgroud remove method retracking..............')
        tic
        rec_image=zeros(dim); rec_image=reshape(rec_image,[],dim(3));
        rec_image(AllBW)=bhvdata_bg(AllBW);
        % rec_image1=reshape(rec_image,dim); implay(rec_image1)
        clear AllBW
        %% tracking to find the binary position of animal
        bw=arrayfun(@(i)...
            rec_image(:,i)>multithresh(rec_image(rec_image(:,i)>prctile(rec_image(:,i),thrP),i)),...
            1:dim(3),'UniformOutput',false);
        bw=cellfun(@(x) imreconstruct(imerode(x,mouseSE),x),bw,'UniformOutput',false);
        bw=cat(2,bw{:}); bw=reshape(bw,dim);
        toc
        clear rec_image
        idx=find(bboxes(:,5)==0);
        disp(['lost frames.......',num2str(length(idx))])
        %% calculate centroids
        [py,px,pt] = ind2sub([bvr,bvc,bvt],find(bw));
        mCenA = NaN(bvt,2);
        mCenA(:,1) = accumarray(pt,px,[],@mean,NaN);
        mCenA(:,2) = accumarray(pt,py,[],@mean,NaN);
        %% save
        movData.mCenA=mCenA;
        movData.bw=bw;
        movData.bboxes=bboxes;
        movData.mazeMask=mazeMask;
        movData.bgImgM=bgImgM;
        movData.dim=dim;
        [filepath,name,ext] = fileparts(tfn);      
        save(fullfile(filepath,[name,'.mat']),'movData','-append')    
        %% plot figure
        bgImgM=reshape(bgImgM,bvr,bvc);
        figure;
        imshow(bgImgM)
        hold on
        plot(mCenA(:,1),mCenA(:,2))
        toc
end
