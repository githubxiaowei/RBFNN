clear;
close all;

ntransient = 100;
nstep = 20000;
tstep0 = 0.01;

% initial_point = [-2.0; -3.7; 20.1];
% LLE = 0.91;
% tstep = tstep0/LLE;
% tspan = [0 ntransient*tstep:tstep:(ntransient+nstep-1)*tstep];
% [T,Y]=ode45(@(t,X) Lorenz(X), tspan, initial_point); 
% T = T(2:end,:); Y = Y(2:end,:);
% figure(1), subplot(4,1,1),plot(1:nstep, Y);
% figure(2), subplot(2,2,1),plot(Y(:,1), Y(:,3));
% figure(3), subplot(2,2,1),plot3(Y(2:end,1), Y(2:end,2), Y(2:end,3))
% csvwrite( 'Lorenz.csv', Y);
% 
% initial_point = [-2; 2;0.2];
% LLE = 0.077;
% tstep = tstep0/LLE;
% tspan = [0 ntransient*tstep:tstep:(ntransient+nstep-1)*tstep];
% [T,Y]=ode45(@(t,X) Rossler(X), tspan, initial_point); 
% T = T(2:end,:); Y = Y(2:end,:);
% figure(1), subplot(4,1,2),plot(1:nstep, Y);
% figure(2), subplot(2,2,2),plot(Y(:,1), Y(:,3));
% figure(3), subplot(2,2,2),plot3(Y(2:end,1), Y(2:end,2), Y(2:end,3))
% csvwrite( 'Rossler.csv', Y);
% 
% 
% initial_point = [-1;0;0.5];
% LLE = 0.20;
% tstep = tstep0/LLE;
% tspan = [0 ntransient*tstep:tstep:(ntransient+nstep-1)*tstep];
% [T,Y]=ode45(@(t,X) RabinovichFabrikant(X), tspan, initial_point); 
% T = T(2:end,:); Y = Y(2:end,:);
% figure(1), subplot(4,1,3),plot(1:nstep, Y);
% figure(2), subplot(2,2,3),plot(Y(:,1), Y(:,3));
% figure(3), subplot(2,2,3),plot3(Y(2:end,1), Y(2:end,2), Y(2:end,3))
% csvwrite( 'RabinovichFabrikant.csv', Y);
% 
% initial_point = [-1;0;0.1];
% LLE = 0.58;
% tstep = tstep0/LLE;
% tspan = [0 ntransient*tstep:tstep:(ntransient+nstep-1)*tstep];
% [T,Y]=ode45(@(t,X) scroll4(X), tspan, initial_point); 
% T = T(2:end,:); Y = Y(2:end,:);
% figure(1), subplot(4,1,4),plot(1:nstep, Y);
% figure(2), subplot(2,2,4),plot(Y(:,1), Y(:,3));
% figure(3), subplot(2,2,4),plot3(Y(2:end,1), Y(2:end,2), Y(2:end,3))
% csvwrite( 'scroll4.csv', Y);

initial_points = [[-2.0; -3.7; 20.1],[-2; 2;0.2],[-1;0;0.5],[-1;0;0.1]];
LLEs = [0.91,0.07,0.20,0.47];
view = [[1,3];[1,2];[2,3];[1,3]];
systems = {@Lorenz, @Rossler, @RabinovichFabrikant, @scroll4};
system_names = {'Lorenz', 'Rossler', 'Rabinovich Fabrikant', 'Four-scroll'};
for i = 1:4
    initial_point = initial_points(:,i);
    %LLE = LLEs(i);
    LLE=1;
    tstep = tstep0/LLE;
    tspan = [0 ntransient*tstep:tstep:(ntransient+nstep-1)*tstep];
    [T,Y]=ode45(@(t,X) systems{i}(X), tspan, initial_point); 
    T = T(2:end,:); Y = Y(2:end,:);
%     Y = (Y - min(Y))./(max(Y) - min(Y))*2-1;
    figure(1), 
        subplot(4,1,i),
        plot(1:nstep, Y);
        index = {'(a)', '(b)','(c)','(d)'};
        ylabel(index{i}, 'Rotation', 0, 'Position',[-500,-0.2,-1])
%         grid on
        box off
        ylim([-1.2,1.2])
        xlim([0, nstep])
        
        if i == 1
            legend('x','y','z')
        end
        if i ~= 4
            xticks([])
        end

           
    figure(2), 
        subplot(2,2,i),
        plot(Y(:,view(i,1)), Y(:,view(i,2)));
        axeslabel = {'x','y','z'};
        xlabel(axeslabel{view(i,1)})
        ylabel(axeslabel{view(i,2)})
        title(index{i})
    
    figure(3), 
        subplot(2,2,i),
        plot3(Y(2:end,1), Y(2:end,2), Y(2:end,3))
        grid on
        xlabel('x');
        ylabel('y');
        zlabel('z');
        title(index{i})
    figure(4), 
        subplot(2,2,i),
        plot3(Y(2:2000,1), Y(2:2000,2), Y(2:2000,3))
        grid on
        xlabel('x');
        ylabel('y');
        zlabel('z');
        title(index{i})
%     figure,
%         subplot(3,1,1)
%         autocorr(Y(:,1), 100)
%         subplot(3,1,2)
%         autocorr(Y(:,2), 100)
%         subplot(3,1,3)
%         autocorr(Y(:,3), 100)
   csvwrite([system_names{i},'_raw', '.csv'], Y);
end

function dX = Lorenz(X) 
a = 10;
b = 28;
c = 8/3;
x=X(1); 
y=X(2); 
z=X(3);
dX = zeros(3,1);
dX(1)=a*(y-x);
dX(2)=x*(b-z)-y;
dX(3)=x*y-c*z;
end 

function dX = Rossler(X) 
a = 0.2;
b = 0.2;
c = 5.7; 
x=X(1); 
y=X(2); 
z=X(3);
dX = zeros(3,1);
dX(1) = -y-z;
dX(2) = x+a*y;
dX(3) = b+z*(x-c);
end 

function dX = scroll4(X) 
a = 1.46;
b = 9;
c = 5;
d = 0.06;
x=X(1); 
y=X(2); 
z=X(3);
dX = zeros(3,1);
dX(1)= a * (x - y) - y*z;
dX(2)= -b*y + x*z;
dX(3)= -c*z + d*x + x*y;
end

function dX = RabinovichFabrikant(X) 
a = 1.1;
b = 0.87;
x=X(1); 
y=X(2); 
z=X(3);
dX = zeros(3,1);
dX(1)= y * (z - 1 + x * x) + b * x;
dX(2)= x * (3 * z + 1 - x * x) + b * y;
dX(3)= -2 * z * (a + x * y);
end
