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
disp(sum(optimal_actions(:, 1) ~= 1));
cd ("G:\matlab\bin");
