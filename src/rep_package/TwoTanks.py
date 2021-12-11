#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 18:59:15 2019

@author: wizard1993
"""
import numpy as np
from numpy import sqrt
class TwoTanks:

    def __init__(self,nonLinearInputChar=False):
        self.nonLinearInputChar=nonLinearInputChar
        self.stateSize=2
        self.input_size=1        
        self.inputStateLinearity= True
        self.outputLinearity=  True
        self.exponent=0.5;
        
    def stateMap(self,x,u):   
        if self.nonLinearInputChar:
            x1=x[0]-.5*sqrt(x[0])+.4*np.sign(u[0])*(u[0])**2
        else:
            x1=x[0]-.5*sqrt(x[0])+.4*u[0];
            
        x2=x[1]+.2*sqrt(x[0])-.3*sqrt(x[1]);
        x=np.zeros((self.stateSize,1))
        if  x1>0:
            x[0]=x1
        else:
            x[0]=0
        
        if x2>0:
            x[1]=x2
        else:
            x[1]=0       
    
        return x
    
    def outputMap(self,xk):
        return np.array(xk[1])
    
    def systemDynamics(self,dim, flag=True):
        x_k=np.ones((self.stateSize,1))
        
        y_n=np.zeros((dim,1))
        u_n=np.zeros((dim,self.input_size))
        #u_n=0
        noise=np.random.normal(1,1.0,size=(self.input_size,dim))        
        if  flag:            
            print('a')
        else:
            print('b')
        u=np.array([ 0.0])
        
        for i in range(0,dim):        
            if i%10000==0:
                print('.',  end='')        

            u[0]=noise[0][int(i/5)]
            
            y_n[i]=self.outputMap(x_k)*1
            x_k=self.stateMap(x_k,np.reshape(u,(self.input_size,1)))*1
            #u+=np.random.normal(0,0.05,size=(self.input_size))
            u_n[i]=u

        return y_n,u_n
    
    
    def loop(self,x_k,duk):
        y_n=np.array([])
        
        for i in range(0,len(duk)):
            u=np.reshape(np.array(duk[i]),(1,1))
            
            u=u*self.stdU+self.meanU
            #print(u)
            temp=self.outputMap(x_k);
            temp=(temp-self.meanY)/self.stdY
            y_n=np.append(y_n,temp)
            
            #print(temp)
            x_k=self.stateMap(x_k,u)                            
        
        return y_n*1,x_k*1
    
    
    def prepareDataset(self,sizeT,sizeV):
        y_n,u_n=self.systemDynamics(sizeT,True)       
        y_Vn,u_Vn=self.systemDynamics(sizeV,True)       
        self.meanY=np.mean(y_n)
        self.meanU=np.mean(u_n)
        self.stdY=np.std(y_n)
        self.stdU=np.std(u_n)
        y_n=(y_n-self.meanY)/self.stdY+np.random.normal(0,0.02,(sizeT,1))
        y_Vn=(y_Vn-self.meanY)/self.stdY+np.random.normal(0,0.02,(sizeV,1))
        u_n=(u_n-self.meanU)/self.stdU+np.random.normal(0,0.02,(sizeT,1))
        u_Vn=(u_Vn-self.meanU)/self.stdU+np.random.normal(0,0.02,(sizeV,1))
        print(y_n.shape)
        print(u_n.shape)
        print(y_Vn.shape)
        print(u_Vn.shape)
        return np.reshape(u_n,(sizeT,1)),np.reshape(y_n,(sizeT,1)),\
                    np.reshape(u_Vn,(sizeV,1)),np.reshape(y_Vn,(sizeV,1))
        
