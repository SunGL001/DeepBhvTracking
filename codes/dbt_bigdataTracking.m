% This is an intergrated  approach for tracking the position of lab animals for bigdata video of batch files. 
% Please be familiar with steps for 'dbt_batchTracking' before using 'dbt_bigdataTracking'.
% Import the video path, load the corresponding detector, and define the tracking area manually. 
% Finaly, the bounding-box of animal can be tracked by YOLO and the center of animal can be calculated by background substration in the bounding-box automatically. 
% Caution: background is highly correlated with color of target with background (lines 13 and 164-168).
% Developed by GuangLong Sun et al., 2021.
% Please cite: DeepBhvTracking: A Novel Behavior Tracking Method for Laboratory Animals Based on Deep Learning
clear; close all; clc
%% set neccessary parameters (this part needs to be set manually before tracking)
fPath='D:\DeepBhvTracking'; % videos root path
fns=struct2table(dir(fullfile(fPath,'**','*.mp4'))); % Support MOV, MP4, avi, hd5 et,al.
fd='D:\DeepBhvTracking\demo_detecor_BlackMice.mat'; % detector path
animalColor = 1;  % '1' means black animal removing background and '2' means white animal. details are shown in lines 164-168.
bin = 5000; % patch data tracking by bins
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
    % for i=1:2     % number of simlilar items
    % Y2=drawrectangle;
    % mazeMask2{i} = Y2.createMask;
    % end
    % delete(Y2)
    % close
    % mazeMask2=logical(sum(cat(3,mazeMask2{:}),3));
    % mazeMask=(mazeMask & ~mazeMask2);
    %% save
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
load(fd)
thrP = 85; mouseWidth = 2; % default
mouseSE = strel('disk',round(mouseWidth/2));
%% tracking start
disp ('Batch tracking start:.............')
for kk=1:nfns
    %  try
    disp(['tracking_file......',num2str(kk),'......total...',num2str(nfns)])
    tfn=fns{kk};
    [filepath,name,ext] = fileparts(tfn);
    mtfn=[fullfile(filepath,name),'.mat'];
    mat=matfile(mtfn);
    movData=mat.movData;   mazeMask=movData.mazeMask;
    try
        bboxe=movData.bboxes;
    catch
        %% get file
        tic
        [bhvdata,~]=dbt_bhvread(tfn,loadvar);
        toc
        %% adjust demension,if demension less than 4, adjust to 4 demension
        dim=size(bhvdata);
        if length(dim)<4
            bhvdata=repmat(bhvdata,[1 1 1 3]);
            bhvdata=permute(bhvdata,[1 2 4 3]);
        end
        %% bin data
        nbin=ceil(dim(end)/bin);
        Sdata=cell(nbin,1);
        for i=1:nbin
            idx=(i-1)*bin+1:i*bin;
            idx(idx>dim(end))=[];
            Sdata{i}=bhvdata(:,:,:,idx);
        end
        clear bhvdata
        %% patch data tracking by bins
        bboxes2=cell(nbin,1);
        tic
        disp('detcection by deep learning ..........................')
        parfor m=1:nbin
            bhvdata=Sdata{m};
            dim1=size(bhvdata);
            bhvdata=reshape(bhvdata,[dim1(1)*dim1(2),dim1(3:4)]);
            bhvdata(~mazeMask,:,:)=255;
            bhvdata=reshape(bhvdata,dim1);
            % implay(bhvdata)
            %% detect position
            [bboxes,scores] = arrayfun(@(i) detect(detector,bhvdata(:,:,:,i),'Threshold',0.2),1:size(bhvdata,4),'UniformOutput',false);
            %% correct multi target
            bboxes1=cellfun(@(x,y) [x(y==max(y),:) max(y)],bboxes, scores, 'UniformOutput', false);
            for i=1:length(bboxes1)
                temp=bboxes1{i};
                if isempty(temp)
                    bboxes1{i}=zeros(1,5);
                end
            end
            tbboxes2=double(cat(1,bboxes1{:}));
            bboxes2{m}=tbboxes2;
        end
        toc
        % save(tfn,'bboxes2','-append')
        clear Sdata
        %% use background tracking method to redefine the position
        tic
        [bhvdata,~]=dbt_bhvread(tfn,loadvar);
        toc
        if length(size(bhvdata)) > 3
            bhvdata=arrayfun(@(i) {rgb2gray(bhvdata(:,:,:,i))},1:size(bhvdata,4));
            bhvdata=cat(3,bhvdata{:});
        end
        bgImgM = uint8(mean(bhvdata,3));
        %  figure
        %  imshow(bgImgM)
        bgImgM=bgImgM(:);
        %% seperate data again
        Sdata=cell(nbin,1);
        for i=1:nbin
            idx=(i-1)*bin+1:i*bin;
            idx(idx>dim(end))=[];
            Sdata{i}=bhvdata(:,:,idx);
        end
        clear bhvdata
        %% denosie tracking
        mCenA=cell(nbin,1);
        bw=cell(nbin,1);
        tic
        for m=1:nbin
            disp('background method to redefine  center')
            disp('Data load ..........................')
            bhvdata=Sdata{m};
            dim=size(bhvdata); [bvr,bvc,bvt]=size(bhvdata);
            bhvdata=reshape(bhvdata,[],size(bhvdata,3));
            %% create mask
            aR =bboxes2{m};
            aR =aR(:,1:4);
            aR(:,3:4)=aR(:,1:2)+aR(:,3:4);
            aR1=aR(:,[1:2 3 2 3 4 1 4 1:2]);
            AllBW=arrayfun(@(b)  poly2mask(aR1(b,1:2:end),aR1(b,2:2:end),dim(1),dim(2)),1:size(aR1,1),'UniformOutput',false);
            AllBW=cat(3,AllBW{:}); AllBW=reshape(AllBW,[],dim(3));
            % AllBW=reshape(AllBW,dim); implay(AllBW)            
            %% image remove background
            if animalColor==1
            bhvdata_bg=bgImgM-bhvdata; % for dark mice remove background
            elseif animalColor==2
            bhvdata_bg=bhvdata-bgImgM; % for white mice remove background   
            end
            bhvdata_bg1=reshape(bhvdata_bg,dim);
            %implay(bhvdata_bg1)
            %  clear bhvdata
            %% rectangel imaging
            disp('backgroud remove method retracking..............')
            rec_image=zeros(dim); rec_image=reshape(rec_image,[],dim(3));
            rec_image(AllBW)=bhvdata_bg(AllBW);
            %  clear AllBW
            %% tracking to find the binary position of animal
            tbw=arrayfun(@(i)...
                rec_image(:,i)>multithresh(rec_image(rec_image(:,i)>prctile(rec_image(:,i),thrP),i)),...
                1:dim(3),'UniformOutput',false);
            tbw=cellfun(@(x) imreconstruct(imerode(x,mouseSE),x),tbw,'UniformOutput',false);
            tbw=cat(2,tbw{:}); tbw=reshape(tbw,dim);            
            % clear rec_image
            %% calculate centroids
            [py,px,pt] = ind2sub([bvr,bvc,bvt],find(tbw));
            tmCenA = NaN(bvt,2);
            tmCenA(:,1) = accumarray(pt,px,[],@mean,NaN);
            tmCenA(:,2) = accumarray(pt,py,[],@mean,NaN);
            mCenA{m}=tmCenA;
            bw{m}=tbw;
        end
        toc
        clear tbw
        %% cat
        mCenA=cat(1,mCenA{:});
        bw=cat(3,bw{:});
        bboxes2=cat(1,bboxes2{:});
        idx=find(bboxes2(:,5)==0);
        disp(['lost frames.......',num2str(length(idx))])
        %% save
        movData.mCenA=mCenA;
        movData.bw=bw;
        movData.bboxes=bboxes2;
        movData.mazeMask=mazeMask;
        movData.bgImgM=bgImgM;
        dim=size(bw); movData.dim=dim;
        [filepath,name,ext] = fileparts(tfn);
        save(fullfile(filepath,[name,'.mat']),'movData','-append')
        %% plot figure
        bgImgM=reshape(bgImgM,dim(1),dim(2));
        figure;
        imshow(bgImgM)
        hold on
        plot(mCenA(:,1),mCenA(:,2))
        toc
    end
end

