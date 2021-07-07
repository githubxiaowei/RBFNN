clc;clear;close all;
[x,y]=meshgrid(linspace(-2,2));
h=streamslice(x,y, -y-x.*(x.^2+y.^2-1) , x -y.*(x.^2+y.^2-1) );
% h=streamslice(x,y, -y+x.*(x.^2+y.^2-1) , x +y.*(x.^2+y.^2-1) );
% h=streamslice(x,y, -y , x  );
xlabel('x');ylabel('y');
set(h,'Color','k')
axis equal
hold on
theta=0:pi/30:2*pi;
x1=cos(theta);y1=sin(theta);
plot(x1,y1,'r')
xlim([-2,2]);
ylim([-2,2]);
set(gca,'XTick',[-2,-1,0,1,2],'YTick',[-2,-1,0,1,2])

set(gcf, 'unit', 'centimeters', 'position',[10 10 10 10]);
set(gcf,'Units','Inches');
pos = get(gcf,'Position');
set(gcf,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
print(gcf, 'circle_stable.pdf', '-dpdf','-r600')

figure,
t = linspace(0,100,10000)
plot(t,sin(t)),hold on,
plot(t,cos(t))
legend('x', 'y')
xlabel('t')
box off
set(gca,'Units','normalized', 'Position',[0.05 0.2 0.9 0.7])
set(gcf, 'unit', 'centimeters', 'position',[10 10 20 5]);
set(gcf,'Units','Inches');
pos = get(gcf,'Position');
set(gcf,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
print(gcf, 'circle_observe.pdf', '-dpdf','-r600')

% ntransient = 1;
% nstep = 20000;
% initial_point = [10; 0];
% tstep = 0.01;
% tspan = [0 ntransient*tstep:tstep:(ntransient+nstep-1)*tstep];
% [T,Y]=ode45(@LC, tspan, initial_point); 
% % T = T(2:end,:); 
% % Y = Y(2:end,:);
% 
% plot(T,Y);
% hold on;
% initial_point = [-8; 8];
% tstep = 0.01;
% tspan = [0 ntransient*tstep:tstep:(ntransient+nstep-1)*tstep];
% [T,Y]=ode45(@LC, tspan, initial_point); 
% % T = T(2:end,:); 
% % Y = Y(2:end,:);
% plot(T,Y);
% 
% 
% data = csvread( ['matlab_gendata/Lorenz_raw.csv']);
% 
% 
% function dX = LC(t, X) 
% x=X(1); 
% y=X(2); 
% dX = zeros(2,1);
% dX(1)=-y - x.*(x.^2+y.^2-1) +sin(2*t);
% dX(2)=x - y.*(x.^2+y.^2-1);
% end
% 
% function dX = Lorenz_drive(t, X) 
% a = 10;
% b = 28;
% c = 8/3;
% x=X(1); 
% y=X(2); 
% z=X(3);
% 
% x = data(t,1)
% dX = zeros(3,1);
% dX(1)=a*(y-x);
% dX(2)=x*(b-z)-y;
% dX(3)=x*y-c*z;
% end 