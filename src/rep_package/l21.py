#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 22:41:38 2019

@author: wizard1993
"""
import numpy as np

import keras

from keras.layers import Input, Dense
from keras.models import Model

from keras.regularizers import Regularizer,l1
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import time
from keras import backend as K


class l21(Regularizer):
    """Regularizer for L21 regularization.
    # Arguments
        C: Float; L21 regularization factor.
    """

    def __init__(self, C=0.0,a=0,b=0,bias=0.000):
        self.a=a
        self.b=b
        C=K.cast_to_floatx(C)
        # self.C = (bias+C)*np.concatenate([0+a-np.array(range(0,a)),0+b-np.array(range(0,b))])        
        self.C = (bias+C)*np.square(np.concatenate([0+a-np.array(range(0,a)),0+b-np.array(range(0,b))])); print("****Squared weigthing enabled****")
        print(self.C)

    def __call__(self, x):
#        print(self.C)
#        print("=>"+str(x))
#        print("=>"+str(K.sum(K.abs(x),1)))
        print(x)
        w=K.sum(K.abs(x),1)
        print(str(w))
        w=w[0:self.a+self.b];
        print(w*self.C)
        # print("=>"+str(K.transpose(w)))
        return K.sum(w*self.C)
        
        

    def get_config(self):
        return {'C': float(self.l1)}
