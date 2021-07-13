function [bhv_video,behaviorFrameRate]=dbt_bhvread(fn,varargin)
[~,~,ext]=fileparts(fn);
if ~isempty(varargin) && strcmp(ext,'.mat')  
    mat=matfile(fn);
    bhv_video=mat.(varargin{1});
    behaviorFrameRate=[];
    if length  (varargin  )>1
        behaviorFrameRate=mat.(varargin{2});
   end
elseif strcmp(ext,'.tif') || strcmp(ext,'.tiff')
    info=imfinfo(fn,'tiff');
    nframe=length(info);
    %     width=info(1).Width;
    %     height=info(1).Height;
    tic
    bhv_video=arrayfun(@(i) {imread(fn,i)},1:nframe);
    bhv_video=cat(3,bhv_video{:});
    toc
    behaviorFrameRate=[];
elseif strcmp(ext,'.hdf5') || strcmp(ext,'.h5') 
    info = h5info(fn);
    var=info.Datasets.Name;
    bhv_video = h5read(fn,var);
      behaviorFrameRate=[];
else
    obj=VideoReader(fn);
    behaviorFrameRate=obj.FrameRate;
    tic
    bhv_video=importdata(fn);
    toc
    if isstruct(bhv_video)
        bhv_video=struct2table(bhv_video);
        bhv_video=bhv_video.cdata;
        bhv_video=cat(4,bhv_video{:});
    end
end
