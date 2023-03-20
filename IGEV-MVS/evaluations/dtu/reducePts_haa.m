function [ptsOut,indexSet] = reducePts_haa(pts, dst)

%Reduces a point set, pts, in a stochastic manner, such that the minimum sdistance
% between points is 'dst'. Writen by abd, edited by haa, then by raje

nPoints=size(pts,2);

indexSet=true(nPoints,1);
RandOrd=randperm(nPoints);

%tic
NS = KDTreeSearcher(pts');
%toc

% search the KNTree for close neighbours in a chunk-wise fashion to save memory if point cloud is really big
Chunks=1:min(4e6,nPoints-1):nPoints;
Chunks(end)=nPoints;

for cChunk=1:(length(Chunks)-1)
    Range=Chunks(cChunk):Chunks(cChunk+1);
    
    idx = rangesearch(NS,pts(:,RandOrd(Range))',dst);
    
    for i = 1:size(idx,1)
        id =RandOrd(i-1+Chunks(cChunk));
        if (indexSet(id))
            indexSet(idx{i}) = 0;
            indexSet(id) = 1;
        end
    end
end

ptsOut = pts(:,indexSet);

disp(['downsample factor: ' num2str(nPoints/sum(indexSet))]);
