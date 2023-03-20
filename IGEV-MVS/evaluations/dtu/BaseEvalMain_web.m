clear all
close all
format compact
clc

% script to calculate distances have been measured for all included scans (UsedSets)

dataPath='D:\xgw\IterMVS_data\MVS Data\';
plyPath='/data/xgw/IGEV_MVS/conf_03/';
resultsPath='/data/xgw/IGEV_MVS/outputs_conf_03/';


method_string='itermvs';
light_string='l3'; % l3 is the setting with all lights on, l7 is randomly sampled between the 7 settings (index 0-6)
representation_string='Points'; %mvs representation 'Points' or 'Surfaces'

switch representation_string
    case 'Points'
        eval_string='_Eval_'; %results naming
        settings_string='';
end

% get sets used in evaluation
UsedSets=[1 4 9 10 11 12 13 15 23 24 29 32 33 34 48 49 62 75 77 110 114 118];

result = zeros(length(UsedSets),4);

dst=0.2;    %Min dist between points when reducing

for cIdx=1:length(UsedSets)
    %Data set number
    cSet = UsedSets(cIdx)
    %input data name
    DataInName=[plyPath sprintf('%s%03d_%s%s.ply',lower(method_string),cSet,light_string,settings_string)]
    
    %results name
    %concatenate strings into one string
    EvalName=[resultsPath method_string eval_string num2str(cSet) '.mat']

    disp(EvalName)
    
    %check if file is already computed
    if(~exist(EvalName,'file'))
        disp(DataInName);
        
        time=clock;time(4:5), drawnow
        
        tic
        Mesh = plyread(DataInName);
        Qdata=[Mesh.vertex.x Mesh.vertex.y Mesh.vertex.z]';
        toc
        
        BaseEval=PointCompareMain(cSet,Qdata,dst,dataPath);
        
        disp('Saving results'), drawnow
        toc
        save(EvalName,'BaseEval');
        toc
        
        % write obj-file of evaluation
%         BaseEval2Obj_web(BaseEval,method_string, resultsPath)
%         toc
        time=clock;time(4:5), drawnow
    
        BaseEval.MaxDist=20; %outlier threshold of 20 mm
        
        BaseEval.FilteredDstl=BaseEval.Dstl(BaseEval.StlAbovePlane); %use only points that are above the plane 
        BaseEval.FilteredDstl=BaseEval.FilteredDstl(BaseEval.FilteredDstl<BaseEval.MaxDist); % discard outliers
    
        BaseEval.FilteredDdata=BaseEval.Ddata(BaseEval.DataInMask); %use only points that within mask
        BaseEval.FilteredDdata=BaseEval.FilteredDdata(BaseEval.FilteredDdata<BaseEval.MaxDist); % discard outliers
        
        fprintf("mean/median Data (acc.) %f/%f\n", mean(BaseEval.FilteredDdata), median(BaseEval.FilteredDdata));
        fprintf("mean/median Stl (comp.) %f/%f\n", mean(BaseEval.FilteredDstl), median(BaseEval.FilteredDstl));
        result(cIdx,1) = mean(BaseEval.FilteredDdata);
        result(cIdx,2) = median(BaseEval.FilteredDdata);
        result(cIdx,3) = mean(BaseEval.FilteredDstl);
        result(cIdx,4) = median(BaseEval.FilteredDstl);
    else
        
        
        load(EvalName);

        BaseEval.MaxDist=20; %outlier threshold of 20 mm
        
        BaseEval.FilteredDstl=BaseEval.Dstl(BaseEval.StlAbovePlane); %use only points that are above the plane 
        BaseEval.FilteredDstl=BaseEval.FilteredDstl(BaseEval.FilteredDstl<BaseEval.MaxDist); % discard outliers
    
        BaseEval.FilteredDdata=BaseEval.Ddata(BaseEval.DataInMask); %use only points that within mask
        BaseEval.FilteredDdata=BaseEval.FilteredDdata(BaseEval.FilteredDdata<BaseEval.MaxDist); % discard outliers
        
        fprintf("mean/median Data (acc.) %f/%f\n", mean(BaseEval.FilteredDdata), median(BaseEval.FilteredDdata));
        fprintf("mean/median Stl (comp.) %f/%f\n", mean(BaseEval.FilteredDstl), median(BaseEval.FilteredDstl));
        result(cIdx,1) = mean(BaseEval.FilteredDdata);
        result(cIdx,2) = median(BaseEval.FilteredDdata);
        result(cIdx,3) = mean(BaseEval.FilteredDstl);
        result(cIdx,4) = median(BaseEval.FilteredDstl);
    end
end

mean_result=mean(result);
fprintf("final evaluation result on all scans: acc.: %f, comp.: %f, overall: %f\n", mean_result(1), mean_result(3), (mean_result(1)+mean_result(3))/2);


