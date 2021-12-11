clear
load(fullfile(matlabroot, 'toolbox', 'ident', 'iddemos', 'data', 'mrdamper.mat'));
U=V(1:2000)
U_val=V(2001:end)
Y=F(1:2000)
Y_val=F(2001:end)
save('Magneto.mat','U','Y','U_val','Y_val')
clear

load(fullfile(matlabroot, 'toolbox', 'ident', 'iddemos', 'data', 'robotarmdata'));
U=ue
U_val=uv1
Y=ye
Y_val=yv1
save('RobotArm.mat','U','Y','U_val','Y_val')

clear

load twotankdata
U=u(1:2000)
U_val=u(2001:end)
Y=y(1:2000)
Y_val=y(2001:end)
save('TwoTanksMatlab.mat','U','Y','U_val','Y_val')



clear
load SNLS80mV.mat
fs=1e7/2^14;
U=V1(30550:38500)';
Y=V2(30550:38500)';
load Schroeder80mV
U_val=V1(10585:10585+1023)';
Y_val=V2(10585:10585+1023)';
save('Silverbox.mat','U','Y','U_val','Y_val')