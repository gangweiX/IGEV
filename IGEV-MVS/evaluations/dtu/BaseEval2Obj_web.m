function BaseEval2Obj_web(BaseEval,method_string,outputPath)

if(nargin<3)
    outputPath='./';
end

% tresshold for coloring alpha channel in the range of 0-10 mm
dist_tresshold=10;

cSet=BaseEval.cSet;

Qdata=BaseEval.Qdata;
alpha=min(BaseEval.Ddata,dist_tresshold)/dist_tresshold;

fid=fopen([outputPath method_string '2Stl_' num2str(cSet) ' .obj'],'w+');

for cP=1:size(Qdata,2)
    if(BaseEval.DataInMask(cP))
        C=[1 0 0]*alpha(cP)+[1 1 1]*(1-alpha(cP)); %coloring from red to white in the range of 0-10 mm (0 to dist_tresshold)
    else
        C=[0 1 0]*alpha(cP)+[0 0 1]*(1-alpha(cP)); %green to blue for points outside the mask (which are not included in the analysis)
    end
    fprintf(fid,'v %f %f %f %f %f %f\n',[Qdata(1,cP) Qdata(2,cP) Qdata(3,cP) C(1) C(2) C(3)]);
end
fclose(fid);

disp('Data2Stl saved as obj')

Qstl=BaseEval.Qstl;
fid=fopen([outputPath 'Stl2' method_string '_' num2str(cSet) '.obj'],'w+');

alpha=min(BaseEval.Dstl,dist_tresshold)/dist_tresshold;

for cP=1:size(Qstl,2)
    if(BaseEval.StlAbovePlane(cP))
        C=[1 0 0]*alpha(cP)+[1 1 1]*(1-alpha(cP)); %coloring from red to white in the range of 0-10 mm (0 to dist_tresshold)
    else
        C=[0 1 0]*alpha(cP)+[0 0 1]*(1-alpha(cP)); %green to blue for points below plane (which are not included in the analysis)
    end
    fprintf(fid,'v %f %f %f %f %f %f\n',[Qstl(1,cP) Qstl(2,cP) Qstl(3,cP) C(1) C(2) C(3)]);
end
fclose(fid);

disp('Stl2Data saved as obj')