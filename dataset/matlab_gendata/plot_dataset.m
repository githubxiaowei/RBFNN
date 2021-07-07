clear;
close all;

system_names = {'Lorenz', 'Rossler', 'Rabinovich Fabrikant', 'Chua'};
N = size(system_names, 2);

for i = 1:N
    Y = csvread( [system_names{i}, '.csv']);
    figure,
    hs = tight_subplot(3, 1, [0.05, 0], [0.2, 0.05], [0.07, 0.07]);
    % [gap_h gap_w] [lower upper] [left right]
    index = {'x', 'y', 'z'};
    train_indice = 1:10000
    val_indice = 10000:15000
    test_indice = 15000:20000
    for j = 1:3
        axes(hs(j));
        plot( train_indice,Y(train_indice,j), 'k'), hold on
        plot( val_indice, Y(val_indice,j), 'b');
        plot( test_indice, Y(test_indice,j), 'r');
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
    set(gcf, 'unit', 'centimeters', 'position',[10 10 15 5])
    set(gcf,'Units','Inches');
    pos = get(gcf,'Position');
    set(gcf,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)]);
    print(gcf, [system_names{i},'_dataset_colored.pdf'], '-dpdf','-r300');
end