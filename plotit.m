function plotit(targets, outputs, Name)

    errors=targets-outputs;
    MSE=mean(errors.^2);
    RMSE=sqrt(MSE);
    error_mean=mean(errors);
    error_std=std(errors);
    subplot(2,2,[1 2]);
    plot(targets,'-.',...
    'LineWidth',1,...
    'MarkerSize',10,...
    'Color',[0.0,0.9,0.0]);
    hold on;
    plot(outputs,'--',...
    'LineWidth',2,...
    'MarkerSize',10,...
    'Color',[0.9,0.0,0.9]);
    legend('Target','Output');
    title(Name);
    xlabel('Samples');
    grid on;
    subplot(2,2,3);
    plot(errors,':',...
    'LineWidth',1.5,...
    'MarkerSize',3,...
    'Color',[0.4,0.1,0.1]);
    title(['MSE = ' num2str(MSE) ', RMSE = ' num2str(RMSE)]);
    ylabel('Errors');
    grid on;
    subplot(2,2,4);
    h=histfit(errors, 80);
    h(1).FaceColor = [.9 .8 .5];
    h(2).Color = [.2 .9 .2];
    title(['Error Mean = ' num2str(error_mean) ', Error StD = ' num2str(error_std)]);
end