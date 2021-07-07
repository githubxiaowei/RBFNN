clear;
close all;

ntransient = 100;
nstep = 20000;
tstep0 = 0.01;
initial_points = [[-2.0; -3.7; 20.1],[-2; 2;0.2],[-1;0;0.5],[1; 0.0; -1]];
systems = {@Lorenz, @Rossler, @RabinovichFabrikant, @chua};
system_names = {'Lorenz', 'Rossler', 'RabinovichFabrikant', 'Chua'};

% for i = 1:4
%     initial_point = initial_points(:,i);
%     %LLE = LLEs(i);
%     LLE=1;
%     tstep = tstep0/LLE;
%     tspan = [0 ntransient*tstep:tstep:(ntransient+nstep-1)*tstep];
%     [T,Y]=ode45(@(t,X) systems{i}(X), tspan, initial_point); 
%     T = T(2:end,:); Y = Y(2:end,:);
%     %Y = (Y - min(Y))./(max(Y) - min(Y))*2-1;
%     figure,
%     hs = tight_subplot(3, 1, [0.05, 0], [0.2, 0.05], [0.07, 0.05]);
%     % [gap_h gap_w] [lower upper] [left right]
%     index = {'x', 'y', 'z'};
%     for j = 1:3
%         axes(hs(j));
%         [pks, locs] = findpeaks(Y(:,j));
% %         appt = mean(locs(2:end) - locs(1:end-1))
%         plot(linspace(tstep0,nstep*tstep0,nstep), Y(:,j), 'k');
%         ylabel(index{j} , 'units', 'normalized' ,'position', [-0.04, 0.5]);
%         xlim([0, nstep*tstep0]);
%         if j ~= 3
%             xticks([])
%         else
%             xlabel('time')
%         end
%     
%          % A4 21*28.5
%         set(gca,'FontSize', 8)
%         box off 
%     end
%     set(gcf, 'unit', 'centimeters', 'position',[10 10 15 5])
%     set(gcf,'Units','Inches');
%     pos = get(gcf,'Position');
%     set(gcf,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
%     print(gcf, [system_names{i},'_components.pdf'], '-dpdf','-r0')        
% end

% 
% nstep = 5000;
% for i = 1
%     initial_point = initial_points(:,i);
%     %LLE = LLEs(i);
%     LLE=1;
%     tstep = tstep0/LLE;
%     tspan = [0 ntransient*tstep:tstep:(ntransient+nstep-1)*tstep];
%     [T,Y]=ode45(@(t,X) systems{i}(X), tspan, initial_point); 
%     T = T(2:end,:); Y = Y(2:end,:);
%     %Y = (Y - min(Y))./(max(Y) - min(Y))*2-1;
%     figure,
%     hs = tight_subplot(3, 1, [0.05, 0], [0.2, 0.05], [0.07, 0.05]);
%     % [gap_h gap_w] [lower upper] [left right]
%     index = {'x', 'y', 'z'};
%     for j = 1:3
%         axes(hs(j));
%         [pks, locs] = findpeaks(Y(:,j));
% 
% %         appt = mean(locs(2:end) - locs(1:end-1))
%    
%         plot(linspace(tstep0,nstep*tstep0,nstep), Y(:,j), 'k');
%         hold on,
%         plot(locs*tstep0, pks, 'Marker','o', 'Color', 'r', 'MarkerSize',3, 'LineStyle', 'none');     
%         ylabel(index{j} , 'units', 'normalized' ,'position', [-0.04, 0.5])
%         
%         xlim([0, nstep*tstep0])
%         if j ~= 3
%             xticks([])
%         else
%             xlabel('time')
%         end
%     
%          % A4 21*28.5
%         set(gca,'FontSize', 8)
%         box off
%       
%     end
%  
%     set(gcf, 'unit', 'centimeters', 'position',[10 10 15 5])
%     set(gcf,'Units','Inches');
%     pos = get(gcf,'Position');
%     set(gcf,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
%     print(gcf, [system_names{i},'_peaks.pdf'], '-dpdf','-r0')   
% end

nstep = 10000;
stepsize = [0.008, 0.059, 0.025, 0.017];

for i = 1:4
    initial_point = initial_points(:,i);

    tstep = stepsize(i);
    tspan = [0 ntransient*tstep:tstep:(ntransient+nstep-1)*tstep];
    [T,Y]=ode45(@(t,X) systems{i}(X), tspan, initial_point); 
    T = T(2:end,:); Y = Y(2:end,:);
    Y = (Y - min(Y))./(max(Y) - min(Y))*2-1;
    figure,
    hs = tight_subplot(3, 1, [0.05, 0], [0.2, 0.05], [0.07, 0.3]);
    % [gap_h gap_w] [lower upper] [left right]
    index = {'x', 'y', 'z'};
    for j = 1:3
        axes(hs(j));
        plot( Y(:,j), 'k');
        ylabel(index{j} , 'units', 'normalized' ,'position', [-0.04, 0.5]);
%         xlim([0, nstep*tstep]);
        if j ~= 3
            xticks([])
        else
            xlabel('step')
        end
    
         % A4 21*28.5
        set(gca,'FontSize', 8)
        box off 
    end
    
    axes, plot3(Y(2:end,1), Y(2:end,2), Y(2:end,3), 'k')
%      xlabel('x'),ylabel('y'),zlabel('z'),
    view(40, 20); grid on
    set(gca, 'position', [0.75 0.2 0.2 0.7])
    set(gcf, 'unit', 'centimeters', 'position',[10 10 15 5])
    set(gcf,'Units','Inches');
    pos = get(gcf,'Position');
    set(gcf,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)])
    print(gcf, [system_names{i},'_components_normalized.pdf'], '-dpdf','-r300')
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

function dX = chua(X) 
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
