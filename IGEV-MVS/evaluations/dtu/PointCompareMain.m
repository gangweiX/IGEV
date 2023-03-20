function BaseEval=PointCompareMain(cSet,Qdata,dst,dataPath)
% evaluation function the calculates the distantes from the reference data (stl) to the evalution points (Qdata) and the
% distances from the evaluation points to the reference

tic
% reduce points 0.2 mm neighbourhood density
Qdata=reducePts_haa(Qdata,dst);
toc

StlInName=[dataPath '/Points/stl/stl' sprintf('%03d',cSet) '_total.ply'];

StlMesh = plyread(StlInName);  %STL points already reduced 0.2 mm neighbourhood density
Qstl=[StlMesh.vertex.x StlMesh.vertex.y StlMesh.vertex.z]';

%Load Mask (ObsMask) and Bounding box (BB) and Resolution (Res)
Margin=10;
MaskName=[dataPath '/ObsMask/ObsMask' num2str(cSet) '_' num2str(Margin) '.mat'];
load(MaskName)

MaxDist=60;
disp('Computing Data 2 Stl distances')
Ddata = MaxDistCP(Qstl,Qdata,BB,MaxDist);
toc

disp('Computing Stl 2 Data distances')
Dstl=MaxDistCP(Qdata,Qstl,BB,MaxDist);
disp('Distances computed')
toc

%use mask
%From Get mask - inverted & modified.
One=ones(1,size(Qdata,2));
Qv=(Qdata-BB(1,:)'*One)/Res+1;
Qv=round(Qv);

Midx1=find(Qv(1,:)>0 & Qv(1,:)<=size(ObsMask,1) & Qv(2,:)>0 & Qv(2,:)<=size(ObsMask,2) & Qv(3,:)>0 & Qv(3,:)<=size(ObsMask,3));
MidxA=sub2ind(size(ObsMask),Qv(1,Midx1),Qv(2,Midx1),Qv(3,Midx1));
Midx2=find(ObsMask(MidxA));

BaseEval.DataInMask(1:size(Qv,2))=false;
BaseEval.DataInMask(Midx1(Midx2))=true; %If Data is within the mask

BaseEval.cSet=cSet;
BaseEval.Margin=Margin;         %Margin of masks
BaseEval.dst=dst;               %Min dist between points when reducing
BaseEval.Qdata=Qdata;           %Input data points
BaseEval.Ddata=Ddata;           %distance from data to stl
BaseEval.Qstl=Qstl;             %Input stl points
BaseEval.Dstl=Dstl;             %Distance from the stl to data

load([dataPath '/ObsMask/Plane' num2str(cSet)],'P')
BaseEval.GroundPlane=P;         % Plane used to destinguise which Stl points are 'used'
BaseEval.StlAbovePlane=(P'*[Qstl;ones(1,size(Qstl,2))])>0; %Is stl above 'ground plane'
BaseEval.Time=clock;            %Time when computation is finished




