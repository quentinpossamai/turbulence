#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 18:59:15 2019

@author: wizard1993
"""
import numpy as np
from numpy import sqrt,cosh,tanh
from scipy.integrate import solve_ivp

# It's actually the hammerstein-wiener! The class is called "LinearSystem" as the the I/O non linearities are a later addition.
 
class LinearSystem:
    
    def __init__(self,nonLinearInputChar=True,satValue=1000,sigma=0.02):
        self.nonLinearInputChar=nonLinearInputChar
        self.U=-1;
        self.sigma=sigma;
        self.satValue=satValue    
        self.flag=True
        self.stateSize=5
        self.input_size=1                
        self.A=np.array([[0.7555,   -0.1991,   0.00000 ,  0.00000  ,0*-0.00909],
                           [0.25000,   0.00000,   0.00000,   0.00000,   0*0.03290],
                           [0.00000,  -0.00000,   0.00000,   0.00000,   0*0.29013],
                           [0.00000,   0.00000,   .00000,   0.00000,  0*-1.05376],
                           [0.00000,   0.00000,   0.00000,   .00000,   0*1.69967]]).T
#        self.B=-np.array([[0.71985,   0.57661,  1.68733,  2.14341,   1.00000]]).T
        self.B=np.array([[-0.5,   0.,  0, 0,   0]]).T
        self.C=np.array([[  0.6993 ,  -0.4427,  0,   0,   0 ,]])
#        
   
    
    def stateMap(self,xk,u):
    
#        if sum(abs(u-np.clip(u,-1000,1000)))>0.01:
#            print('.',end='')
#            pass
        
        u=np.clip(u,-self.satValue,self.satValue)
        
#        if u<-.6 and self.nonLinearInputChar:
#            u=u*1.2
        if u>0 and self.nonLinearInputChar:
            u=np.sqrt(u)
            print('.',end='')
#        print(u)
            
        xk=np.dot(self.A,xk)+np.dot(self.B,u)        
        return xk
        
        
    def outputMap(self,xk):
        #print(np.dot(self.C,xk))
        y=np.dot(self.C,xk)
        return y+5*np.sin(y)*int(self.nonLinearInputChar)
    
    def systemDynamics(self,dim, flag=True,U=-1):
        x_k=np.ones((self.stateSize,1))
        
        y_n=np.zeros((dim,1))
        u_n=np.zeros((dim,self.input_size))
        #u_n=0
        noise=np.random.normal(0,1,size=(self.input_size,dim))        
        if  flag:            
            print('a')
        else:
            print('b')
        u=np.array([ 0.0])
        
        for i in range(0,dim):        
            if i%10000==0:
                print('.',  end='')        
            u[0]=noise[0][int(i/7)]
            uSat=np.clip(u[0],-self.satValue,self.satValue)
            
            y_n[i]=self.outputMap(x_k)*1
            x_k=self.stateMap(x_k,np.reshape(uSat,(self.input_size,1))) *1       
            #u+=np.random.normal(0,0.05,size=(self.input_size))
            u_n[i]=u[0]
#            u_n[i]=uSat

        return y_n,u_n
    
    
    def loop(self,x_k,duk):
        y_n=np.array([])
        x_k=np.reshape(x_k,(self.stateSize,1))
        for i in range(0,len(duk)):
            u=np.array(duk[i])+np.random.normal(0,self.sigma,(1,1))*.0
            u=u*self.stdU+self.meanU
#            print(u)
            temp=self.outputMap(x_k);
#            print(temp)
            temp=(temp-self.meanY)/self.stdY
            y_n=np.append(y_n,temp)+np.random.normal(0,self.sigma,(1,1))*.0
            
            #print(temp)
            x_k=self.stateMap(x_k,u)*1
        
        return y_n[0]*1,x_k
    
    
    def prepareDataset(self,sizeT,sizeV):
        y_n,u_n=self.systemDynamics(sizeT,True)       
        y_Vn,u_Vn=self.systemDynamics(sizeV,True)       
        self.meanY=np.mean(y_n)
        self.meanU=np.mean(u_n)
        self.stdY=np.std(y_n)
        self.stdU=np.std(u_n)
        y_n=(y_n-self.meanY)/self.stdY+np.random.normal(0,self.sigma,(sizeT,1))
        y_Vn=(y_Vn-self.meanY)/self.stdY+np.random.normal(0,self.sigma,(sizeV,1))
        u_n=(u_n-self.meanU)/self.stdU+np.random.normal(0,self.sigma,(sizeT,1))
        u_Vn=(u_Vn-self.meanU)/self.stdU+np.random.normal(0,self.sigma,(sizeV,1))
        print(y_n.shape)
        print(u_n.shape)
        print(y_Vn.shape)
        print(u_Vn.shape)
        return np.reshape(u_n,(sizeT,1)),np.reshape(y_n,(sizeT,1)),\
                    np.reshape(u_Vn,(sizeV,1)),np.reshape(y_Vn,(sizeV,1))
        
