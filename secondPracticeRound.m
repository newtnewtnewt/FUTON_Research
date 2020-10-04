cd ("G:\matlab\bin");
load('transitionr');
load('qldata3');
load('transitionr2');
cd ("G:\JuptyerScripts\Komoroski\MDPtoolbox");
r3=zeros(state_count+2,state_count+2,nact); 
r3(state_count+1,:,:)=-100; 
r3(state_count+2,:,:)=100;
R=sum(transitionr.*r3);
R=squeeze(R);   %remove 1 unused dimension
[~,~,~,~,Qon] = mdp_policy_iteration_with_Q(transitionr2, R, gamma, ones(state_count+2,1));
[~,OptimalAction]=max(Qon,[],2);  %deterministic 
optimal_actions(:,1)=OptimalAction; %save optimal actions
cd ("G:\matlab\bin");
disp('#### OFF-POLICY EVALUATION - MIMIC TRAIN SET ####')
 
% create new version of QLDATA3
r=[100 -100];
r2=r.*(2*(1-Y90)-1); 
qldata=[blocs idx actionbloctrain Y90 zeros(numel(idx),1) r2(:,1) ptid];  % contains bloc / state / action / outcome&reward     %1 = died
qldata3=zeros(floor(size(qldata,1)*1.2),8); 

c=0;
abss=[state_count+2 state_count+1]; %absorbing states numbers
 
        for i=1:size(qldata,1)-1
            c=c+1;
              qldata3(c,:)=qldata(i,[1:3 5 7 7 7 7]);
            if qldata(i+1,1)==1 %end of trace for this patient
                c=c+1;
                qldata3(c,:)=[qldata(i,1)+1 abss(1+qldata(i,4)) 0 qldata(i,6) 0 0 0 qldata(i,7)]; 
            end
        end
        qldata3(c+1:end,:)=[];
        
% add pi(s,a) and b(s,a)
p=0.01; %softening policies  
softpi=physpol; % behavior policy = clinicians' 

for i=1:750
    % For each state, grab all actions that are equal to zero
    ii=softpi(i,:)==0;    
    z=p/sum(ii);   
    nz=p/sum(~ii);    
    softpi(i,ii)=z;   
    softpi(i,~ii)=softpi(i,~ii)-nz;
end
softb=abs(zeros(752,25)-p/24); %"optimal" policy = target policy = evaluation policy 

for i=1:750
     softb(i,OptimalAction(i))=1-p;
end

for i=1:size(qldata3,1)  %adding the probas of policies to qldata3
    if qldata3(i,2)<=750
        qldata3(i,5)=softpi(qldata3(i,2),qldata3(i,3));
        qldata3(i,6)=softb(qldata3(i,2),qldata3(i,3));
        qldata3(i,7)=OptimalAction(qldata3(i,2));   %optimal action
    end
end

qldata3train=qldata3;

tic
 [ bootql,bootwis ] = offpolicy_multiple_eval_010518( qldata3,physpol, 0.99,0,6,750);
toc