
from functools import partial
import tensorflow as tf
import random as rn
from scipy import optimize
import numpy as np
import sys
import time
import matplotlib.pyplot as plt
from DynamicalSystem import LinearSystem
import scipy.io.matlab
from TwoTanks import TwoTanks
from DummyModel import DummyModel



import matplotlib
matplotlib.rcParams["figure.figsize"] = [8, 6]
matplotlib.rcParams["lines.linewidth"] = 2
matplotlib.rcParams["figure.dpi"] = 100
matplotlib.rcParams["font.size"]=14
matplotlib.rcParams["text.usetex"]=True

from enum import Enum,unique
np.random.seed(1)
import keras
from keras import backend as K
from ANNmodel import AdvAutoencoder,datasetLoadUtility

#%% Options
@unique
class systemSelectorEnum(Enum):
    def loadFromDataset(filename,nonLinearInputChar=False):
        dynamicModel=DummyModel();
        dsLoading=datasetLoadUtility();
        Uvero, Yvero, UV,YV=dsLoading.loadDatasetFromMATfile(filename)        
        numel=Uvero.shape[0];
        numelV=UV.shape[0];
        u_n=np.reshape(Uvero.T[0],(numel,1))
        y_n=np.reshape(Yvero.T[0],(numel,1))
        u_Vn=np.reshape(UV.T[0],(numelV,1))
        y_Vn=np.reshape(YV.T[0],(numelV,1))        
        meanY=np.mean(y_n)
        meanU=np.mean(u_n)
        stdY=np.std(y_n)
        stdU=np.std(u_n)
        y_n=(y_n-meanY)/stdY;#+np.random.normal(0,0.05,(sizeT,1))
        y_Vn=(y_Vn-meanY)/stdY;#+np.random.normal(0,0.05,(sizeV,1))
        u_n=(u_n-meanU)/stdU;#+np.random.normal(0,0.05,(sizeT,1))
        u_Vn=(u_Vn-meanU)/stdU;#+np.random.normal(0,0.05,(sizeV,1))        
        return dynamicModel,u_n,y_n,u_Vn,y_Vn
    def MAGNETOdataset():
        return systemSelectorEnum.loadFromDataset('Magneto.mat');
    def TANKSdataset():
        return systemSelectorEnum.loadFromDataset('TwoTanksMatlab.mat');
    def SILVERBOXdataset():
        return systemSelectorEnum.loadFromDataset('Silverbox.mat');
        
    def TWOTANKS(nonLinearInputChar=False):
        dynamicModel=TwoTanks(nonLinearInputChar=Option.nonLinearInputChar);
        U, Y, U_val, Y_val=dynamicModel.prepareDataset(20000,1000)
        return dynamicModel,U,Y,U_val,Y_val
    def BILINEAR(nonLinearInputChar=False): # It's actually the hammerstein-wiener! But the old name stuck
        dynamicModel=LinearSystem(nonLinearInputChar=Option.nonLinearInputChar);
        U, Y, U_val, Y_val=dynamicModel.prepareDataset(10000,1000)
        return dynamicModel,U,Y,U_val,Y_val


class Options():
    def __init__(self):    
        self.nonLinearInputChar=True; 
        self.dynamicalSystemSelector=systemSelectorEnum.TWOTANKS
        self.stringDynamicalSystemSelector=str(self.dynamicalSystemSelector).replace('<function systemSelectorEnum.','').split(' at ')[0]
        self.affineStruct=True;
        self.openLoopStartingPoint=15
        self.horizon=5# for MPC
        self.TRsteps=1
        self.fitHorizon=5# 5 or 2 #it's +1 wrt to paper
        self.n_a=10;#n_a=n_b
        self.useGroupLasso=False;             
        self.stateReduction=True;
        self.regularizerWeight=0.0001;
        self.closedLoopSim=True
        self.enablePlot=True
        self.stateSize=6;
        pass
    
Option=Options()
#%% Parameter parsing
print('para',sys.argv)
if len(sys.argv)>2:    
    Option.fitHorizon=int(sys.argv[2])
    print(int(sys.argv[2]))
    
if len(sys.argv)>3:    
    if int(sys.argv[3])==1:    
        Option.dynamicalSystemSelector=systemSelectorEnum.TWOTANKS
    elif  int(sys.argv[3])==2:
        Option.dynamicalSystemSelector=systemSelectorEnum.BILINEAR  # It's actually the hammerstein-wiener! But the old name stuck
    elif  int(sys.argv[3])==3:
        Option.dynamicalSystemSelector=systemSelectorEnum.MAGNETOdataset
        Option.closedLoopSim=False
    elif  int(sys.argv[3])==4:
        Option.dynamicalSystemSelector=systemSelectorEnum.TANKSdataset
        Option.closedLoopSim=False
    elif  int(sys.argv[3])==5:
        Option.dynamicalSystemSelector=systemSelectorEnum.SILVERBOXdataset
        Option.closedLoopSim=False
        
    Option.stringDynamicalSystemSelector=str(Option.dynamicalSystemSelector).replace('<function systemSelectorEnum.','').split(' at ')[0]
    print(int(sys.argv[3]))

if len(sys.argv)>4:    
    if int(sys.argv[4])==1:
        Option.nonLinearInputChar=True
    else:
        Option.nonLinearInputChar=False
    print(int(sys.argv[4]))
    
if len(sys.argv)>5:    
    Option.stateSize=int(sys.argv[5])
    print(int(sys.argv[5]))
    
if len(sys.argv)>6:        
    Option.n_a=int(sys.argv[6])
    print(float(sys.argv[6]))
    
if len(sys.argv)>7:        
    if int(sys.argv[7])==1:    
        Option.affineStruct=True
    else:
        Option.affineStruct=False
    print(float(sys.argv[7]))

if len(sys.argv)>8:        
    if int(sys.argv[8])==1:    
        Option.affineStruct=False
        Option.useGroupLasso=True;             
        Option.stateReduction=True;
        Option.regularizerWeight=0.0003
    elif int(sys.argv[8])==2:    
        Option.useGroupLasso=True;             
        Option.affineStruct=False
        Option.stateReduction=not True;
        Option.regularizerWeight=0.0003
    else:
        Option.useGroupLasso=False;   
        Option.regularizerWeight=0.0001
        pass
    print(float(sys.argv[8]))
    

import warnings
warnings.filterwarnings("ignore")

    

#%% DS generation and model learning
simulatedSystem,U_n,Y_n,U_Vn,Y_Vn=Option.dynamicalSystemSelector()

model=AdvAutoencoder(affineStruct=Option.affineStruct,useGroupLasso=Option.useGroupLasso,
                     stateReduction=Option.stateReduction,fitHorizon=Option.fitHorizon,
                     strideLen=Option.n_a,#n_a=n_b
                     outputWindowLen=2,#+1 wrt the paper
                     n_layer=3,n_neurons=30,                     
                     regularizerWeight=Option.regularizerWeight,stateSize=Option.stateSize)
model.setDataset(U_n.copy(),Y_n.copy(),U_Vn.copy(),Y_Vn.copy())


inputU,inputY=model.prepareDataset()
model.trainModel()
predictedLeft,stateLeft,oneStepAheadPredictionError,forwardedPredictedError,forwardError=model.model.predict([inputY,inputU])


#%% Functions definition
def prepareMatrices(uSequence,x0):
    logY=[]
    logX=[]
    uSequence=np.array(uSequence)
    
    for u in uSequence:
        u=np.reshape(u,(1,1))
        x0=model.bridgeNetwork.predict([u,x0])   
        y=model.outputEncoder.predict([x0[0]])        
        logY+=[y]
        logX+=[x0]
        x0=x0[0]
        pass
    
    return logX,logY
    pass


def costFunction(uSequence,r,um1,logAB,logC,x0):
    
    logY=[]
    uSequence=np.array(uSequence)
    um1=np.array(um1)
    i=0
    for u in uSequence:
        #u=np.reshape(u,(1,1))
        asda=np.concatenate([x0.ravel(),u.ravel()])
        asda=np.reshape(asda,(Option.stateSize+1,1))
        x0=np.dot(logAB[i][1],asda)
        x0=x0+np.reshape(logAB[i][2],(Option.stateSize,1))
        y=np.dot(logC[i][1],x0)
        logY+=[y[0][-1]]
        i=i+1
        pass
#    logY+=[y[0][1]]
    logY=np.array(logY)
#    print(logY-r)
    cost=.001*np.sum(np.square(uSequence))+\
        .01*np.sum(np.square(uSequence[1:]-uSequence[:-1]))+\
        .01*np.sum(np.square(uSequence[0]-um1))+\
        np.sum(np.square(logY-r))*1
    
    return cost


def evaluateFeatureImportance():
    from matplotlib.ticker import MaxNLocator


    if not Option.stateReduction:
        w=model.convEncoder.get_layer('enc00').get_weights()
        ax = plt.figure(figsize=[8,2]).gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        neuronsCount=np.sum(abs(w[0])>1e-3,1);
#        print(len(neuronsCount))
        windowsLen=int(len(neuronsCount)/2)
        yAxis=range(0,windowsLen)[::-1]
        print(neuronsCount,'encoder=>')    
        plt.title('$encoder$')
        plt.step(yAxis,neuronsCount[0:windowsLen],where='mid')
        plt.step(yAxis,neuronsCount[windowsLen:],where='mid')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.tight_layout() 
        
    else:
        w1=model.bridgeNetwork.get_layer('bridge00').get_weights()
        w=model.outputEncoder.get_layer('dec00').get_weights()
        neuronsCount=np.sum(abs(w1[0][0:-1])>1e-3,1);
        yAxis=range(0,len(neuronsCount))
        print(neuronsCount,'bridge=>')  
        plt.figure(figsize=[8,2])
        plt.title('$bridge$')
        plt.step(yAxis,neuronsCount,where='mid')
        plt.tight_layout() 
        neuronsCount=np.sum(abs(w[0])>1e-3,1);        
        print(neuronsCount,'decoder=>')    
        yAxis=range(0,len(neuronsCount))
        ax = plt.figure(figsize=[8,2]).gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.title('$decoder$')
        plt.step(yAxis,neuronsCount,where='mid')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        plt.tight_layout() 
    pass


def openLoopValidation(validationOnMultiHarmonic=True, reset=-1,YTrue=None,U_Vn=None):    
    openLoopStartingPoint=Option.openLoopStartingPoint
    pastY=np.zeros((model.strideLen,1))
    pastU=np.zeros((model.strideLen,1))
    if YTrue is None:
        x0RealSystem=np.zeros((simulatedSystem.stateSize,))
        
    x0=model.convEncoder.predict([pastY.T,pastU.T])
    logY=[]
    logU=[]
    logYR=[]
    finalRange=1000;
    if not(YTrue is None):
        finalRange=YTrue.shape[0]
    for i in range(0,finalRange):
        u=0.5*np.array([[np.sin(i/(20+0.01*i))]])+0.5
        if not validationOnMultiHarmonic:
            u=[U_Vn[i]]
        if YTrue is None:
            y_kReal,x0RealSystem_=simulatedSystem.loop(x0RealSystem,u)
            x0RealSystem=np.reshape(x0RealSystem_,(simulatedSystem.stateSize,))
        else:
            y_kReal=YTrue[i]
            u=[U_Vn[i]]
            pass 
        
        
        pastU=np.reshape(np.append(pastU,u)[1:],(model.strideLen,1))
        pastY=np.reshape(np.append(pastY,y_kReal)[1:],(model.strideLen,1))
        if i<openLoopStartingPoint or (i%reset==0 and reset>0):
            x0=model.convEncoder.predict([pastY.T,pastU.T])
            print('*',end='')
        else:        
            x0=model.bridgeNetwork.predict([u,x0])[0]
        y=model.outputEncoder.predict([x0])[0]
        if i>=openLoopStartingPoint:
            logY+=[y[0][-2]]
            logYR+=[y_kReal[0]]
            logU+=[u[0]]
      
        print('.',end='')
        
        pass
    print('\n')
    logY=np.array(logY)
    logYR=np.array(logYR)    
    a=np.linalg.norm(np.array(logY)-np.array(logYR))
    b=np.linalg.norm(np.mean(np.array(logYR))-np.array(logYR))
    fit=1-(a/b)
    NRMSE=1-np.sqrt(np.mean(np.square(np.array(logY)-np.array(logYR))))/(np.max(logYR)-np.min(logYR))
    fit=np.max([0,fit])
    NRMSE=np.max([0,NRMSE])
    print('fit: ',fit)
    print('NRMSE: ',NRMSE)
    if Option.enablePlot:
        plt.figure()
        plt.title('open loop simulation from k='+str(openLoopStartingPoint)+" fit="+str(fit))
        y,=plt.plot(logY)
        yr,=plt.plot(logYR)
        et,=plt.plot(np.array(logY)-np.array(logYR))
        plt.tight_layout() 
        plt.legend([y,yr,et],['$\hat y$','$y_{real}$','estimation error'])
    return fit,NRMSE,logY,logYR

#%% Model Validation Validation
validationOnMultiHarmonic=[True,False]
reset=[1,10,-1]
for r in reset:
    for voM in validationOnMultiHarmonic:        
        start = time.time()
        YtrueToPass=None
        if 'dataset' in Option.stringDynamicalSystemSelector:
            YtrueToPass=Y_Vn.copy()
        fit,NRMSE,logY,logYR=openLoopValidation(validationOnMultiHarmonic=voM,reset=r,YTrue=YtrueToPass,U_Vn=U_Vn.copy())
        end= time.time()
        print('elapsed time in simulation:',end-start)
        print("validationOnMultiHarmonic:",voM,end=' ')
        print("reset every:",r,end=' ')
        print('fit: ',fit,' NRMSE: ',NRMSE)

#%% Closed loop Simulation with MPC       
u=[U_Vn[0]]

if Option.closedLoopSim and Option.affineStruct:
    logY=[]
    logU=[]
    logYR=[]
    MPCHorizon=Option.horizon
    pastY=np.zeros((model.strideLen,1))
    pastU=np.zeros((model.strideLen,1))
    x0RealSystem=np.zeros((simulatedSystem.stateSize,))
    x0=model.convEncoder.predict([pastY.T,pastU.T])
    bounds=[(-.8,.8) for i in range(0,MPCHorizon)]
#    bounds=[(-1,1) for i in range(0,MPCHorizon)]
    pastRes=np.ones((MPCHorizon))*0;
    start = time.time()
    logY+=[0]
    for i in range(0,400):   
        x0=model.convEncoder.predict([pastY.T,pastU.T])
        r=[ 0.7*np.array([[np.sin(j/(20+0.01*j))]])+.7 for j in range(i,i+MPCHorizon)]
#        if i>200:
#            r=1+r*0
#        else:
#            r=-1+r*0
#        r=np.array([[.5+1.5*np.sin(i/(50+i/100))]])
    #    r=0.5*np.array([[np.sin(i/(20+0.01*i))]])+1
    #    r=np.array([[1.5+np.sin(i/(50+i/100))]])
        logY+=[r[0][0]]
        for _ in range(0,Option.TRsteps):
            logAB,logC=prepareMatrices(pastRes,x0)
            lamdaCostFunction=lambda x: costFunction(x,r,u[0][0],logAB,logC,x0)
            result=optimize.minimize(lamdaCostFunction,pastRes,bounds=bounds)
            u=np.array(result.x[0]).reshape((1,1))
            pastRes=result.x
        pastRes[0:-1]=pastRes[1:]
        # pastRes[-1]=0
        y_kReal,x0RealSystem=simulatedSystem.loop(x0RealSystem,u)    
        x0RealSystem=x0RealSystem.copy()
        pastU=np.reshape(np.append(pastU,u)[1:],(model.strideLen,1))
        pastY=np.reshape(np.append(pastY,y_kReal)[1:],(model.strideLen,1))        
        logYR+=[y_kReal[0]]
        logU+=[u[0]]
        print('.',end='')    
        pass
    end= time.time()
    print('\n')
    if Option.enablePlot:
        plt.figure()
        plt.title('Closed loop simulaton')
        uP,=plt.plot(logU)
        plt.grid()
        yP,=plt.plot(logYR)
        rP,=plt.plot(logY)
        plt.tight_layout() 
        plt.legend([uP,yP,rP],['$u_k$','$y_k$','$r_k$'])
    print('elapsed time in MPC:',end-start)
#print(fit)
#%% Feature Importance
if Option.useGroupLasso:
    if Option.affineStruct:
        print("******WARNING: affine struct is enabled******")
    print("evaluating state importance=>"+str(Option.stateReduction))
    evaluateFeatureImportance();


#%% These functions are used to generate plots for the paper
def prettyPrintStatsUseNA(aOutput,aInput):
    aOutput=np.array(aOutput)
    aInput=np.array(aInput)
    xAxis=range(0,aInput.shape[1])[::-1]
    from matplotlib.ticker import MaxNLocator
    ax = plt.figure(figsize=[8,2]).gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    #Notice: they are inverted with respect to the output of the feature importance function
    lineOutput,=plt.step(xAxis,aOutput.T,where='mid')    
    lineInput,=plt.step(xAxis,aInput.T,where='mid')        
    plt.legend([lineOutput,lineInput],['$\\{y_k\\}$','$\\{u_k\\}$'])
    plt.grid()    
    plt.xlabel('time-step~delay')
    plt.tight_layout()
def prettyPrintStatsUseNX(ADecoder,aBridge):
    aBridge=np.array(aBridge)
    ADecoder=np.array(ADecoder)
    xAxis=range(1,ADecoder.shape[1]+1)[::-1]
    from matplotlib.ticker import MaxNLocator
    ax = plt.figure(figsize=[8,2]).gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    lineBridge,=plt.step(xAxis,aBridge.T,where='mid')    
    lineDecoder,=plt.step(xAxis,ADecoder.T,where='mid')      
    plt.legend([lineBridge,lineDecoder],['$bridge$','$decoder$'])
    plt.grid()
    plt.xlabel('state~component')
    plt.tight_layout()
    

print(Option.__dict__)
scipy.io.matlab.savemat('dump{0}{1}.mat'.format(Option.stringDynamicalSystemSelector,Option.nonLinearInputChar),{'U':U_n,'Y':Y_n,'U_val':U_Vn,'Y_val':Y_Vn,'Option':str(Option.__dict__)})