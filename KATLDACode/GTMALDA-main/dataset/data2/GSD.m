function [ DGS ] = GSD(LDG)
LDG=load('LD_adjmat.txt');
LDG=LDG.'
%GSD Summary of this function goes here
%   Detailed explanation goes here
    
nd=size(LDG,1);
normSum=0;
for i=1:nd
    
   normSum=normSum+((norm(LDG(i,:),2)).^2);
    
end

rd=1/(normSum/nd);

for i=1:nd
   for j=1:nd
       sub=LDG(i,:)-LDG(j,:);
        DGS(i,j)=exp(-rd*((norm(sub,1)).^2));
       
   end 
end
save('GSD','DGS')
end

