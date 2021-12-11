# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 18:15:37 2018

@author: wizard1993
"""


import scipy


import numpy as np
import time
from scipy import io

from matplotlib import pyplot as plt

import keras

from keras.layers import Input, Dense
from keras.models import Model

from keras.regularizers import Regularizer,l1,l2
from keras.constraints import *
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from keras import backend as K
import tensorflow as tf
from l21 import l21

#

class datasetLoadUtility:
    def __init__(self):
        pass

    def loadDatasetFromMATfile(self, filename='-1'):
        dataset = scipy.io.loadmat(filename)
        U, Y, U_val, Y_val= [dataset.get(x) for x in ['U', 'Y', 'U_val', 'Y_val']]
        return U,Y,U_val,Y_val
        pass
    def loadFieldFromMATFile(selfself, filename,fields):
        dataset = scipy.io.loadmat(filename)
        return [dataset.get(x) for x in fields]


class AdvAutoencoder:
    batch_size = 24;
    stateSize=-1
    nonlinearity = ''
    n_neurons = -1
    validation_split = -1;
    n_a = -1
    n_b = -1
    freeloading=10;
    def __init__(self, nonlinearity='relu', n_neurons=30, n_layer=3,    fitHorizon=5,       
                 useGroupLasso=True,stateReduction=False,
                 validation_split=0.05, stateSize=-1,strideLen=10, outputWindowLen=2,
                 affineStruct=True,
                 regularizerWeight=0.0005):
        self.nonlinearity = nonlinearity
        self.outputWindowLen=outputWindowLen
        self.stateSize=stateSize
        self.n_neurons = n_neurons
        self.stateReduction=stateReduction
        self.validation_split = validation_split
        self.strideLen=strideLen
        self.n_layer=n_layer;
        self.model=None        
        self.affineStruct=affineStruct
        self.MaxRange=fitHorizon
        self.regularizerWeight=regularizerWeight
        self.kernel_regularizer=l2(self.regularizerWeight) 
        self.useGroupLasso=useGroupLasso
        self.shuffledIndexes=None
        self.constraintOnInputHiddenLayer=None;
        if useGroupLasso and regularizerWeight>0.0:
            
            # self.constraintOnInputHiddenLayer=unit_norm();
            if stateReduction:
                self.inputLayerRegularizer=l21(self.regularizerWeight,a=0,b=stateSize) 
                # self.inputLayerRegularizer=l1(self.regularizerWeight) 
            else:
                self.inputLayerRegularizer=l21(self.regularizerWeight,a=strideLen,b=strideLen) 
#                self.inputLayerRegularizer=l1(self.regularizerWeight) 
            self.kernel_regularizer=l2(0.0001); print("=>if GroupLasso is used,  l2 regularizer is set to 0.0001 in all bridge, encoder, and decoder")
            # print("=>if GroupLasso is used,  l2 regularizer is set to the same weigth in all bridge, encoder, and decoder")
        else:
            self.inputLayerRegularizer=self.kernel_regularizer
        
        pass


    def mean_pred(y_true, y_pred):     
        
        return keras.losses.mae(y_pred*0,y_pred)


    def setDataset(self, U, Y, U_val, Y_val):
        if U is not None:
            self.U = U.copy()
            self.N_U = U.shape[1]
            
        if Y is not None:
            self.Y = Y.copy()
            self.N_Y = Y.shape[1]

        if U_val is not None:
            self.U_val = U_val.copy()
        if Y_val is not None:
            self.Y_val = Y_val.copy()
        
        pass

        
    
    
    def encoderNetwork(self,future=0):
        inputs_U = Input(shape=((self.strideLen)*self.N_U,))
        inputs_Y = Input(shape=((self.strideLen)*self.N_Y,))
        inputConcat=keras.layers.concatenate([inputs_Y,inputs_U],name='concatIk') 
#        x=keras.layers.BatchNormalization()(inputConcat)   
        iKR=self.kernel_regularizer
        if self.useGroupLasso and (not self.stateReduction):
            iKR=self.inputLayerRegularizer    
        x = Dense(                       
                kernel_regularizer=iKR,
                kernel_constraint=self.constraintOnInputHiddenLayer,
            units=self.n_neurons, activation=self.nonlinearity,name='enc0'+str(future))(inputConcat)
        for i in range(0,self.n_layer-1):
            x = Dense(
                    use_bias=True,
                    kernel_regularizer=self.kernel_regularizer,
            units=self.n_neurons, activation=self.nonlinearity,name='enc'+str(i+1)+str(future))(x)
        x = Dense(
                kernel_regularizer=self.kernel_regularizer,
            units=self.stateSize, activation="linear",name='encf'+str(future))(x)
        ann = Model(inputs=[inputs_Y,inputs_U], outputs=[x])
        return ann
    
    def decoderNetwork(self,future=0):
        inputs_state = Input(shape=(self.stateSize,))       
#        x=keras.layers.BatchNormalization()(inputs_state)   
        iKR=self.kernel_regularizer
        if self.useGroupLasso and self.stateReduction:
            iKR=self.inputLayerRegularizer    
        x = Dense(           
                kernel_regularizer=iKR,                
                kernel_constraint=self.constraintOnInputHiddenLayer,
            units=self.n_neurons, activation=self.nonlinearity,name='dec0'+str(future))(inputs_state)
        for i in range(0,self.n_layer-1):
            x = Dense(
                    use_bias=True,
                    kernel_regularizer=self.kernel_regularizer,
            units=self.n_neurons, activation=self.nonlinearity,name='dec'+str(i+1)+str(future))(x)
      
        if self.affineStruct:
            x = Dense(
#                kernel_regularizer=self.kernel_regularizer,
            units=self.outputWindowLen*self.stateSize, activation="linear",name='decf'+str(future))(x)
            x=keras.layers.Reshape((self.outputWindowLen,self.stateSize))(x)        
            out=keras.layers.dot([x,inputs_state],axes=-1)
        else:
            out=Dense(
                kernel_regularizer=self.kernel_regularizer,
            units=self.outputWindowLen*self.N_Y, activation="linear",name='decf'+str(future))(x)    
            x=out;
            
        ann = Model(inputs=[inputs_state], outputs=[out,x])
        return ann

    def bridgeNetwork(self,future=0):
        inputs_state = Input(shape=(self.stateSize,),name='inputState')       
        inputs_novelU = Input(shape=(self.N_U,),name='novelInput')      
        inputConcat=keras.layers.concatenate([inputs_state,inputs_novelU]) 
#        x=keras.layers.BatchNormalization()(inputConcat)      
        iKR=self.kernel_regularizer
        if self.useGroupLasso and self.stateReduction:
            iKR=self.inputLayerRegularizer    
            print('using group lasso on state')
        x = Dense(           
                kernel_regularizer=iKR,
                kernel_constraint=self.constraintOnInputHiddenLayer,
            units=self.n_neurons, activation=self.nonlinearity,name='bridge0'+str(future))(inputConcat)
        for i in range(0,self.n_layer-1):
            x = Dense(
                    use_bias=True,
                    kernel_regularizer=self.kernel_regularizer,
            units=self.n_neurons, activation=self.nonlinearity,name='bridge'+str(i+1)+str(future))(x)
        bias = Dense(
                    kernel_regularizer=self.kernel_regularizer,
                units=self.stateSize, activation="linear",name='bridgeBias'+str(future))(x)       
       
        if self.affineStruct:   
            ABunshape = Dense(units=self.stateSize*(self.stateSize+self.N_U), 
                      activation="linear",name='bridgef'+str(future))(x)     
            AB=keras.layers.Reshape((self.stateSize,self.stateSize+self.N_U))(ABunshape)                                
            out=keras.layers.dot([AB,inputConcat],axes=-1)
            out=keras.layers.add([bias,out])
            ann = Model(inputs=[inputs_novelU,inputs_state], outputs=[out,AB,bias])
        else:
            out=bias
            ann = Model(inputs=[inputs_novelU,inputs_state], outputs=[out,x,bias])
        
        
        
        
        return ann    
    
    
   
        
    def ANNModel(self):

        
        inputs_Y = Input(shape=((self.strideLen+self.MaxRange)*self.N_Y,), name="input_y")
        inputs_U = Input(shape=((self.strideLen+self.MaxRange)*self.N_U,), name="input_u")
        
        strideLen=self.strideLen
        bridgeNetwork=self.bridgeNetwork()
        convEncoder=self.encoderNetwork()
        outputEncoder=self.decoderNetwork()
        predictionErrorCollection=[]
        forwardErrorCollection=[]
        forwardedPredictedErrorCollection=[]
        predictedOKCollection=[]
        stateKCollection=[]
        MaxRange=self.MaxRange
        forwardedState= None
        for k in range(0,MaxRange):
            IYk=keras.layers.Lambda(lambda x:x[:,k:strideLen+k])(inputs_Y)         
            IUk=keras.layers.Lambda(lambda x:x[:,k:strideLen+k])(inputs_U)     
            ITargetk=keras.layers.Lambda(lambda x:x[:,strideLen+k-self.outputWindowLen+1:strideLen+k+1])(inputs_Y) 
            novelIUk=keras.layers.Lambda(lambda x:x[:,strideLen+k:strideLen+k+1])(inputs_U) 
            print(ITargetk,'dd')
            stateK=convEncoder([IYk,IUk])
            predictedOK=outputEncoder(stateK)[0]
            predictedOKCollection+=[predictedOK]
            stateKCollection+=[stateK]
            predictionErrork=keras.layers.subtract([predictedOK,ITargetk],name='oneStepDecoderError'+str(k)) 
            predictionErrork=keras.layers.Lambda(lambda x:K.abs(x))(predictionErrork)
            predictionErrorCollection+=[predictionErrork]            
            if not(forwardedState is None):                  
                forwardedStateN=[bridgeNetwork([novelIUk,stateK])[0]]
                for thisF in forwardedState:
                    forwardErrork=keras.layers.subtract([stateK,thisF]) 
                    forwardErrork=keras.layers.Lambda(lambda x:K.abs(x))(forwardErrork)
                    forwardErrorCollection+=[forwardErrork]
                    
                    forwardedPredictedOutputK=outputEncoder(thisF)[0]
                    forwardedPredictedErrork=keras.layers.subtract([forwardedPredictedOutputK,ITargetk])         
                    forwardedPredictedErrorCollection+=[forwardedPredictedErrork]              
                    
                    forwardedStateN+=[bridgeNetwork([novelIUk,thisF])[0]]
                    
                forwardedState=forwardedStateN
                
                    
            else:
                forwardedState=[bridgeNetwork([novelIUk,stateK])[0]]
#            PreviousState=matchedk
            # print(IYk,IUk,ITargetk)
            # print()
        
        oneStepAheadPredictionError=keras.layers.concatenate(predictionErrorCollection,name='oneStepDecoderError')
#        forwardError=keras.layers.concatenate(forwardErrorCollection,name='forwardError') 
        # print('len(oneStepAheadPredictionError) ',oneStepAheadPredictionError.shape)
        if len(forwardedPredictedErrorCollection)>1:
            forwardedPredictedError=keras.layers.concatenate(forwardedPredictedErrorCollection,name='multiStep_decodeError')         
        else:
            forwardedPredictedError=keras.layers.Lambda(lambda x:K.abs(x),name='multiStep_decodeError')(forwardedPredictedErrorCollection[0])         
        
        if len(forwardErrorCollection)>1:            
            forwardError=keras.layers.concatenate(forwardErrorCollection,name='forwardError') 
        else:            
            forwardError=keras.layers.Lambda(lambda x:K.abs(x),name='forwardError')(forwardErrorCollection[0])         
        
#        IYleft=keras.layers.Lambda(lambda x:x[:,0:-2])(inputs_Y)         
#        IYRight=keras.layers.Lambda(lambda x:x[:,1:-1])(inputs_Y)         
#        IUleft=keras.layers.Lambda(lambda x:x[:,0:-2])(inputs_U)         
#        IURight=keras.layers.Lambda(lambda x:x[:,1:-1])(inputs_U) 
#        
#        ITargetLeft=keras.layers.Lambda(lambda x:x[:,-self.outputWindowLen-1:-1])(inputs_Y)         
#        ITargetRight=keras.layers.Lambda(lambda x:x[:,-self.outputWindowLen:])(inputs_Y)         
#        
#        novelIU=keras.layers.Lambda(lambda x:x[:,-2:-1])(inputs_U) 
#        
#        stateLeft=convEncoder([IYleft,IUleft])
#        stateRight=convEncoder([IYRight,IURight])
#        
#        predictedLeft=outputEncoder(stateLeft)[0]
#        predictedRight=outputEncoder(stateRight)[0]
#        predictionErrorLeft=keras.layers.subtract([predictedLeft,ITargetLeft]) 
#        predictionErrorRight=keras.layers.subtract([predictedRight,ITargetRight]) 
#        absErrorRight=keras.layers.Lambda(lambda x:keras.backend.abs(x))(predictionErrorRight) 
#        absErrorLeft=keras.layers.Lambda(lambda x:keras.backend.abs(x))(predictionErrorLeft) 
#
#        oneStepAheadPredictionError=keras.layers.concatenate([absErrorRight,absErrorLeft],name='oneStepAheadPredictionError')
#       
#            
#        
#        matchedRight=bridgeNetwork([novelIU,stateLeft])[0]
#        forwardError=keras.layers.subtract([matchedRight,stateRight],name='forwardError') 
#        forwardedPredictedOutput=outputEncoder(matchedRight)[0]
#        forwardedPredictedError=keras.layers.subtract([forwardedPredictedOutput,ITargetRight],name='forwardedPredictedError')         
#        
        ann = Model(inputs=[inputs_Y,inputs_U], outputs=[predictedOKCollection[0],stateKCollection[0],oneStepAheadPredictionError,forwardedPredictedError,forwardError])     
#        ann = Model(inputs=[inputs_Y,inputs_U], outputs=[predictedLeft,stateLeft,oneStepAheadPredictionError,forwardedPredictedError,forwardError])     
#        
        
        return ann,convEncoder,outputEncoder,bridgeNetwork



    def computeGradients(self,train_stateVector=None,train_inputVector=None,index=0):
        if (train_inputVector is None) or (train_stateVector is None):
            train_stateVector,train_inputVector,train_outputVector= self.prepareDataset(self.U, self.Y)
        
        self.sess=K.get_session()
        
        gr=self.sess.run(self.gradientState,feed_dict={self.model.input[0]: train_stateVector,self.model.input[1]: train_inputVector})
        return gr
    
    def prepareDataset(self, U=None, Y=None):    
        if U is None: U=self.U
        if Y is None: Y=self.Y
        pad=self.MaxRange-2
        strideLen=self.strideLen+pad
        print(strideLen)
        lenDS=U.shape[0]
        inputVector=np.zeros((lenDS-2,self.N_U*(strideLen+2)))
        outputVector=np.zeros((lenDS-2,self.N_Y*(strideLen+2)))       
        offset=self.strideLen+1+pad
        
        for i in range(offset, lenDS):            
            regressor_StateInputs = np.ravel(U[i-strideLen-1:i+1]);
            regressor_StateOutputs = np.ravel(Y[i-strideLen-1:i+1]);
            
            inputVector[i-offset]=regressor_StateInputs.copy()
            outputVector[i-offset]=regressor_StateOutputs.copy()            
            pass

        return inputVector[:i-offset+1].copy(),outputVector[:i-offset+1].copy()


    def trainModel(self, shuffled:bool =True):
        tmp=self.privateTrainModel(shuffled,None,kFPE=.0,kAEPrediction=10,kForward=.3);        
        tmp=self.privateTrainModel(shuffled,tmp,1,0.,10);        
        pass
    def privateTrainModel(self, shuffled:bool =True,tmp=None,kFPE=1,kAEPrediction=1,kForward=1):
        inputVector,outputVector=self.prepareDataset()        
 
        
        optimizer = keras.optimizers.Adam(lr=0.002,amsgrad=True,clipvalue=0.5)
#        optimizer = keras.optimizers.Nadam(lr=0.002)
        if not (tmp is None):
            model = self.model;
            convEncoder=self.convEncoder
#            convEncoder.trainable=False
            outputEncoder=self.outputEncoder
#            outputEncoder.trainable=False            
            bridgeNetwork=self.bridgeNetwork     
            model.set_weights(tmp)
            pass
        else:
            model,convEncoder,outputEncoder,bridgeNetwork = self.ANNModel()
        if self.shuffledIndexes is None:
            self.shuffledIndexes=np.random.permutation(list(range(0,np.shape(outputVector)[0])));
            
        shuffledIndexes=self.shuffledIndexes
        model.compile(optimizer=optimizer, 
                      loss_weights={'multiStep_decodeError':kFPE,'oneStepDecoderError':kAEPrediction,'forwardError':kForward},
                      loss={"multiStep_decodeError": AdvAutoencoder.mean_pred,
                            "oneStepDecoderError": AdvAutoencoder.mean_pred,                            
                            'forwardError': AdvAutoencoder.mean_pred},)
        
        model.fit({'input_y':outputVector[shuffledIndexes,:],'input_u':inputVector[shuffledIndexes,:]}, 
                  {'multiStep_decodeError':outputVector[:,0:5]*0,'oneStepDecoderError':outputVector[:,0:5]*0,
                   'forwardError':outputVector[:,0:5]*0
                   },                                     
                  epochs=150,
                  verbose=1,
                  validation_split=self.validation_split, shuffle=shuffled,
                  batch_size=self.batch_size,
                  callbacks=[ReduceLROnPlateau(factor=0.3, min_delta=0.001, patience=3),
                             EarlyStopping(patience=8, min_delta=0.001,monitor='val_loss'),                           
                             ], )
        self.model = model;
        self.convEncoder=convEncoder
        self.outputEncoder=outputEncoder
        self.bridgeNetwork=bridgeNetwork      
        tmp=model.get_weights()
        return tmp

    
    def getModel(self):
        return self.model
        pass

    def evaluateNetwork(self, U_val,Y_val):
        train_stateVector,train_inputVector,train_outputVector= self.prepareDataset(U_val,Y_val)
        t = time.time()
        fitted_Y = self.model.predict([train_stateVector, train_inputVector])        
        elapsed = time.time() - t
        
        return fitted_Y,train_outputVector,elapsed
    
    def validateModel(self, plot:bool =True):
        fitted_Y,train_outputVector,elapsed=self.evaluateNetwork(self.U_val, self.Y_val)
        fitted_Y=fitted_Y[0]
        if plot:
            plt.figure(figsize=(7, 7))
            plt.plot(fitted_Y)
            plt.plot(train_outputVector)
            plt.show()
            pass
        return fitted_Y,train_outputVector,elapsed
        pass
