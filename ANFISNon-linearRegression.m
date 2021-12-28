%% 

% This code takes AverageLocalizationError (ALE 2021) dataset from input
% and applies ANFIS non-linear regression on columns 3 and 4 of data as
% Inputs. Also, column 5 is label. ANFIS uses Subtractive Clustering.
% Feel free to contact me for this code
% Seyed Muhammad Hossein Mousavi
% mosavi.a.i.buali@gmail.com

% Dataset cite:
% Singh, A., Kotiyal, V., Sharma, S., Nagar, J. and Lee, C.C., 2020. A Machine Learning
% Approach to Predict the Average Localization Error With Applications to Wireless 
% Sensor Networks. IEEE Access, 8, pp.208253-208263.

%%
clc;
clear;
data=load('AverageLocalizationError.txt');
Inputs=data(:,3:4);
Targets=data(:,5);
nData = size(Inputs,1);

%% Shuffle the Data
PERM = randperm(nData); % Permutation to Shuffle Data
%
pTrain=0.7; % Divide Train and Test
nTrainData=round(pTrain*nData);
TrainInd=PERM(1:nTrainData);
TrainInputs=Inputs(TrainInd,:);
TrainTargets=Targets(TrainInd,:);
pTest=1-pTrain;
nTestData=nData-nTrainData;
TestInd=PERM(nTrainData+1:end);
TestInputs=Inputs(TestInd,:);
TestTargets=Targets(TestInd,:);

% Parameters of FIS Generation Methods
DefaultValues=0.2; % Play with this parameter to desire
Radius=DefaultValues;
fis=genfis2(TrainInputs,TrainTargets,Radius);
        
% Training ANFIS Structure
MaxEpoch=200;                
ErrorGoal=0;               
InitialStepSize=0.01;         
StepSizeDecreaseRate=0.09;    
StepSizeIncreaseRate=1.9;    
TrainOptions=[MaxEpoch ...
              ErrorGoal ...
              InitialStepSize ...
              StepSizeDecreaseRate ...
              StepSizeIncreaseRate];

DisplayInfo=true;
DisplayError=true;
DisplayStepSize=true;
DisplayFinalResult=true;
DisplayOptions=[DisplayInfo ...
                DisplayError ...
                DisplayStepSize ...
                DisplayFinalResult];

OptimizationMethod=1;

fis=anfis([TrainInputs TrainTargets],fis,TrainOptions,DisplayOptions,[],OptimizationMethod);

% Apply ANFIS to Data
Outputs=evalfis(Inputs,fis);
TrainOutputs=Outputs(TrainInd,:);
TestOutputs=Outputs(TestInd,:);

% Errors
TrainErrors=TrainTargets-TrainOutputs
TrainMSE=mean(TrainErrors.^2)
TrainRMSE=sqrt(TrainMSE)
TrainErrorMean=mean(TrainErrors)
TrainErrorSTD=std(TrainErrors)
TestErrors=TestTargets-TestOutputs
TestMSE=mean(TestErrors.^2)
TestRMSE=sqrt(TestMSE)
TestErrorMean=mean(TestErrors)
TestErrorSTD=std(TestErrors)

% Plots 
figure;
set(gcf, 'Position',  [50, 50, 1000, 350])
plotit(TrainTargets,TrainOutputs,'Train Data');
figure;
set(gcf, 'Position',  [50, 100, 1000, 350])
plotit(TestTargets,TestOutputs,'Test Data');
figure;
set(gcf, 'Position',  [50, 150, 1000, 350])
plotit(Targets,Outputs,'All Data');

% figure;
% plotregression(TrainTargets, TrainOutputs, 'Train Data')
% figure;
% plotregression(TestTargets, TestOutputs, 'Test Data')
% figure;
% plotregression(Targets, Outputs, 'All')

figure;
set(gcf, 'Position',  [80, 80, 1200, 350])
subplot(1,3,1)
[population2,gof] = fit(TrainTargets,TrainOutputs,'poly4');
plot(TrainTargets,TrainOutputs,'o',...
    'LineWidth',3,...
    'MarkerSize',5,...
    'Color',[0.3,0.9,0.2]);
    title(['Train - R =  ' num2str(1-gof.rmse)]);
        xlabel('Train Target');
    ylabel('Train Output');   
hold on
plot(population2,'b-','predobs');
    xlabel('Train Target');
    ylabel('Train Output');   
hold off
subplot(1,3,2)
[population2,gof] = fit(TestTargets, TestOutputs,'poly4');
plot(TestTargets, TestOutputs,'o',...
    'LineWidth',3,...
    'MarkerSize',5,...
    'Color',[0.3,0.9,0.2]);
    title(['Test - R =  ' num2str(1-gof.rmse)]);
    xlabel('Test Target');
    ylabel('Test Output');    
hold on
plot(population2,'b-','predobs');
    xlabel('Test Target');
    ylabel('Test Output');
% hold off
subplot(1,3,3)
[population2,gof] = fit(Targets,Outputs,'poly4');
plot(Targets,Outputs,'h',...
    'LineWidth',1,...
    'MarkerSize',5,...
    'Color',[0.3,0.6,0.2]);
    title(['All - R =  ' num2str(1-gof.rmse)]);
        xlabel('Target');
    ylabel('Output');   
hold on
plot(population2,'k-','predobs');
    xlabel('Target');
    ylabel('Output');   
hold off

figure;
set(gcf, 'Position',  [500, 300, 600, 350])
gensurf(fis, [1 2], 1, [90 90]);
xlim([min(Inputs(:,1)) max(Inputs(:,1))]);
ylim([min(Inputs(:,2)) max(Inputs(:,2))]);
% Some Metrics
fprintf('The (TrainRMSE) value is = %0.4f.\n',TrainRMSE)
fprintf('The (TrainMSE) value is = %0.4f.\n',TrainMSE)
fprintf('The (TrainErrorMean) value is = %0.4f.\n',TrainErrorMean)
fprintf('The (TrainErrorSTD) value is = %0.4f.\n',TrainErrorSTD)
fprintf('The (TestRMSE) value is = %0.4f.\n',TestRMSE)
fprintf('The (TestMSE) value is = %0.4f.\n',TestMSE)
fprintf('The (TestErrorSTD) value is = %0.4f.\n',TestErrorSTD)
fprintf('The (TestErrorMean) value is = %0.4f.\n',TestErrorMean)
