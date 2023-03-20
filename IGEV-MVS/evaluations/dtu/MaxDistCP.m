function Dist = MaxDistCP(Qto,Qfrom,BB,MaxDist)

Dist=ones(1,size(Qfrom,2))*MaxDist;

Range=floor((BB(2,:)-BB(1,:))/MaxDist);

tic
Done=0;
LookAt=zeros(1,size(Qfrom,2));
for x=0:Range(1),
    for y=0:Range(2),
        for z=0:Range(3),
            
            Low=BB(1,:)+[x y z]*MaxDist;
            High=Low+MaxDist;
            
            idxF=find(Qfrom(1,:)>=Low(1) & Qfrom(2,:)>=Low(2) & Qfrom(3,:)>=Low(3) &...
                Qfrom(1,:)<High(1) & Qfrom(2,:)<High(2) & Qfrom(3,:)<High(3));
            SQfrom=Qfrom(:,idxF);
            LookAt(idxF)=LookAt(idxF)+1; %Debug
            
            Low=Low-MaxDist;
            High=High+MaxDist;
            idxT=find(Qto(1,:)>=Low(1) & Qto(2,:)>=Low(2) & Qto(3,:)>=Low(3) &...
                Qto(1,:)<High(1) & Qto(2,:)<High(2) & Qto(3,:)<High(3));
            SQto=Qto(:,idxT);
            
            if(isempty(SQto))
                Dist(idxF)=MaxDist;
            else
                KDstl=KDTreeSearcher(SQto');
                [~,SDist] = knnsearch(KDstl,SQfrom');
                Dist(idxF)=SDist;
                
            end
            
            Done=Done+length(idxF); %Debug
            
        end
    end
    %Complete=Done/size(Qfrom,2);
    %EstTime=(toc/Complete)/60
    %toc
    %LA=[sum(LookAt==0),...
    %	sum(LookAt==1),...
   % 	sum(LookAt==2),...
   % 	sum(LookAt==3),...
   % 	sum(LookAt>3)]
end

