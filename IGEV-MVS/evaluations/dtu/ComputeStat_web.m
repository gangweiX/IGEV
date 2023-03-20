clear all
close all
format compact
clc

% script to calculate the statistics for each scan given this will currently only run if distances have been measured
% for all included scans (UsedSets)

% modify the path to evaluate your models
dataPath='/home/SampleSet/MVS Data/';
resultsPath='/home/PatchmatchNet/outputs/';

MaxDist=20; %outlier thresshold of 20 mm

time=clock;

method_string='patchmatchnet';
light_string='l3'; %'l7'; l3 is the setting with all lights on, l7 is randomly sampled between the 7 settings (index 0-6)
representation_string='Points'; %mvs representation 'Points' or 'Surfaces'

switch representation_string
    case 'Points'
        eval_string='_Eval_'; %results naming
        settings_string='';
end

% get sets used in evaluation
UsedSets=[1 4 9 10 11 12 13 15 23 24 29 32 33 34 48 49 62 75 77 110 114 118];

nStat=length(UsedSets);

% struct
BaseStat.nStl=zeros(1,nStat);
BaseStat.nData=zeros(1,nStat);
BaseStat.MeanStl=zeros(1,nStat);
BaseStat.MeanData=zeros(1,nStat);
BaseStat.VarStl=zeros(1,nStat);
BaseStat.VarData=zeros(1,nStat);
BaseStat.MedStl=zeros(1,nStat);
BaseStat.MedData=zeros(1,nStat);

for cStat=1:length(UsedSets) %Data set number
    
    currentSet=UsedSets(cStat);
    
    %input results name
    EvalName=[resultsPath method_string eval_string num2str(currentSet) '.mat'];
    
    disp(EvalName);
    load(EvalName);
    
    Dstl=BaseEval.Dstl(BaseEval.StlAbovePlane); %use only points that are above the plane 
    Dstl=Dstl(Dstl<MaxDist); % discard outliers
    
    Ddata=BaseEval.Ddata(BaseEval.DataInMask); %use only points that within mask
    Ddata=Ddata(Ddata<MaxDist); % discard outliers
    
    BaseStat.nStl(cStat)=length(Dstl);
    BaseStat.nData(cStat)=length(Ddata);
    
    BaseStat.MeanStl(cStat)=mean(Dstl);
    BaseStat.MeanData(cStat)=mean(Ddata);
    
    BaseStat.VarStl(cStat)=var(Dstl);
    BaseStat.VarData(cStat)=var(Ddata);
    
    BaseStat.MedStl(cStat)=median(Dstl);
    BaseStat.MedData(cStat)=median(Ddata);
    
    disp("acc");
    disp(mean(Ddata));
    disp("comp");
    disp(mean(Dstl));
    time=clock;
end

disp(BaseStat);
disp("mean acc")
disp(mean(BaseStat.MeanData));
disp("mean comp")
disp(mean(BaseStat.MeanStl));

totalStatName=[resultsPath 'TotalStat_' method_string eval_string '.mat']
save(totalStatName,'BaseStat','time','MaxDist');



