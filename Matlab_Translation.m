MIMICtable = readtable("G:/MIMIC-ALL/MIMIC-PATIENTS/patient_data.csv");
patientdata=table2array(MIMICtable);

mdp_count=500;               % nr of repetitions (total nr models)
clustering_iter=32;            % how many times we do clustering (best solution will be chosen)
cluster_sample=0.25;                 % proportion of the data we sample for clustering
gamma=0.99;                % gamma
transition_threshold=5;              % threshold for pruning the transition matrix
final_policies=1;                 % count of saved policies
state_count=750;                   % nr of states
action_count=5;                     % nr of actions (2 to 10)
crossval_iter=5;                     % nr of crossvalidation runs (each is 80% training / 20% test)
optimal_actions=NaN(752,mdp_count);       % record of optimal actions
model_data=NaN(mdp_count*2,30);  % saves data about each model (1 row per model)
bestmodels_data=cell(mdp_count,15);  % saving best candidate models

% #################   Convert training data and compute conversion factors    ######################

% all 47 columns of interest
colbin = {'gender','mechvent','max_dose_vaso','re_admission'};
colnorm={'age','Weight_kg','GCS','HR','SysBP','MeanBP','DiaBP','RR','Temp_C','FiO2_1',...
    'Potassium','Sodium','Chloride','Glucose','Magnesium','Calcium',...
    'Hb','WBC_count','Platelets_count','PTT','PT','Arterial_pH','paO2','paCO2',...
    'Arterial_BE','HCO3','Arterial_lactate','SOFA','SIRS','Shock_Index','PaO2_FiO2','cumulated_balance'};
collog={'SpO2','BUN','Creatinine','SGOT','SGPT','Total_bili','INR','input_total','input_4hourly','output_total','output_4hourly'};

colbin=find(ismember(MIMICtable.Properties.VariableNames,colbin));colnorm=find(ismember(MIMICtable.Properties.VariableNames,colnorm));collog=find(ismember(MIMICtable.Properties.VariableNames,collog));

% find patients who died in ICU during data collection period
% ii=MIMICtable.bloc==1&MIMICtable.died_within_48h_of_out_time==1& MIMICtable.delay_end_of_record_and_discharge_or_death<24;
% icustayidlist=MIMICtable.icustayid;
% ikeep=~ismember(icustayidlist,MIMICtable.icustayid(ii));
patientdata=table2array(MIMICtable);
% patientdata=patientdata(ikeep,:);
icustayidlist=MIMICtable.icustayid;
icuuniqueids=unique(icustayidlist); %list of unique icustayids from MIMIC
idxs=NaN(size(icustayidlist,1),mdp_count); %record state membership test cohort

MIMICraw=MIMICtable(:, [colbin colnorm collog]);
MIMICraw=table2array(MIMICraw);  % RAW values
MIMICzs=[patientdata(:, colbin)-0.5 zscore(patientdata(:,colnorm)) zscore(log(0.1+patientdata(:, collog)))];  
MIMICzs(:,[4])=log(MIMICzs(:,[ 4])+.6);   % MAX DOSE NORAD 
MIMICzs(:,45)=2.*MIMICzs(:,45);   % increase weight of this variable

stream = RandStream('mlfg6331_64'); options = statset('UseParallel',1,'UseSubstreams',1,'Streams',stream); warning('off','all')
% for modl=1:mdp_count  % MAIN LOOP OVER ALL MODELS
   
N=numel(icuuniqueids); %total number of rows to choose from
grp=floor(crossval_iter*rand(N,1)+1);  %list of 1 to 5 (20% of the data in each grp) -- this means that train/test MIMIC split are DIFFERENT in all the 500 models
crossval=1;
trainidx=icuuniqueids(crossval~=grp);
testidx=icuuniqueids(crossval==grp);
train=ismember(icustayidlist,trainidx);
test=ismember(icustayidlist,testidx);
X=MIMICzs(train,:);
Xtestmimic=MIMICzs(~train,:);
blocs=patientdata(train,1);
bloctestmimic=patientdata(~train,1);
ptid=patientdata(train,2);
ptidtestmimic=patientdata(~train,2);
outcome=10; %   HOSP _ MORTALITY = 8 / 90d MORTA = 10
Y90=patientdata(train,outcome);  
%fprintf('########################   MODEL NUMBER : ');       fprintf('%d \n',modl);         disp( datestr(now))
N=size(X,1); %total number of rows to choose from
sampl=X(find(floor(rand(N,1)+cluster_sample)),:);
[~,C] = kmeans(sampl,state_count,'Options',options,'MaxIter',10000,...
'Start','plus','Display','final','Replicates',clustering_iter);
[idx]=knnsearch(C,X);  %N-D nearest point search: look for points closest to each centroid
disp(size(C))
disp(size(X))



