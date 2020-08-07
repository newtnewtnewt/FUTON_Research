load('transitionr');
load('qldata3');
load('transitionr2');
r3=zeros(state_count+2,state_count+2,nact); 
r3(state_count+1,:,:)=-100; 
r3(state_count+2,:,:)=100;
R=sum(transitionr.*r3);
R=squeeze(R);   %remove 1 unused dimension
life_total = 0;
death_total = 0;
disp(R);
total_rewards = 0;
for i = 1:state_count+2
   for j = 1:nact
       if R(i, j) ~= 0
            total_rewards = total_rewards + 1;
       end
   end
end
disp(total_rewards);
% for i = 1:size(transitionr2, 1)
%         for j = 1:25
%             if transitionr2(i, 751, j) ~= 0
%                 life_total = life_total + 1;
%             end
%             if transitionr2(i, 752, j) ~= 0
%                 death_total = death_total + 1; 
%             end
%         end
% end

% disp(unique(qldata3(qldata3(:, 2) == 751), 1));
% disp(unique(qldata3(qldata3(:, 2) == 752), 1));
% disp(life_total);
% disp(death_total);

% for i = 1:size(R, 1)
%     for j = 1:25
%         if R(i, j) ~= 0
%             count_total = count_total + 1;
%         end
%     end
% end
%disp(count_total);
% disp(size(R));
