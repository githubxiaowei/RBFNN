clear;
close all;

ntransient = 100;
nstep = 20000;

initial_points = [[-2.0; -3.7; 20.1],[-2; 2;0.2],[-1;0;0.5],[1;0;-1]];
stepsize = [0.008, 0.059, 0.025, 0.017];
view = [[1,3];[1,2];[2,3];[1,3]];
systems = {@Lorenz, @Rossler, @RabinovichFabrikant @Chua};
system_names = {'Lorenz', 'Rossler', 'Rabinovich Fabrikant', 'Chua'};
N = size(systems,2)
for i = 1:N
    initial_point = initial_points(:,i);
    tstep = stepsize(i);
    tspan = [0 ntransient*tstep:tstep:(ntransient+nstep-1)*tstep];
    [T,Y]=ode45(@(t,X) systems{i}(X), tspan, initial_point); 
    T = T(2:end,:); 
    Y = Y(2:end,:);
    Y = (Y - min(Y))./(max(Y) - min(Y))*2-1;
    figure(1), 
        subplot(N,1,i),
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
        if i ~= N
            xticks([])
        end
    
    figure(3), 
        subplot(2,2,i),
        plot3(Y(2:end,1), Y(2:end,2), Y(2:end,3))
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
    csvwrite([system_names{i}, '.csv'], Y);
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


function dX = Chua(X) 
a = 12.8;
b = 19.1;
c = 0.6;
d= -1.1;
e= 0.45;
x=X(1); 
y=X(2); 
z=X(3);
dX = zeros(3,1);
dX(1)= a * (y-(c*x + d*x*abs(x) + e*x^3));
dX(2)= x - y + z;
dX(3)= -b*y;
end
