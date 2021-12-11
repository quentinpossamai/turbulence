clear
% load('TwoTanksMatlab.mat')
% load('Silverbox.mat')
% load('dumpNlTankLin.mat')
% load('dumpHW.mat')
% load('Magneto.mat')
load('dumpSILVERBOXdatasetFalse.mat')

meanY=mean(Y)
meanU=mean(U)
stdY=std(Y)
stdU=std(U)

Y_val=(Y_val-meanY)./stdY;
Y=(Y-meanY)./stdY;
U_val=(U_val-meanU)./stdU;
U=(U-meanU)./stdU;
z=iddata(Y,U)
zVal=iddata(Y_val(1:end),U_val(1:end))
horizon=10
collectionFIT=[]
collectionBFR=[]
for iteration=1:10
    ff = feedforwardnet([30 30 30]);
    ff.layers{1}.transferFcn = 'poslin';
    ff.layers{2}.transferFcn = 'poslin';
    ff.layers{3}.transferFcn = 'poslin';
    ff.trainParam.epochs = 500;
    sys = nlarx(z,[10 10 1],neuralnet(ff)); 
    [YH, FIT, X0]=compare(zVal,sys);
    collectionFIT=[collectionFIT;FIT]
    fitCheck=1-norm(YH.OutputData(1:end)-Y_val(1:end),'fro')/norm(mean(Y_val(1:end))-Y_val(1:end),'fro')
    collectionBFR=[collectionBFR;fitCheck]
end
