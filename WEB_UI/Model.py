

import os
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv1D,BatchNormalization
from keras.layers import Activation,Dropout,Dense,Flatten,MaxPooling1D
from keras.layers import LSTM,Bidirectional,TimeDistributed,RepeatVector
import pandas as pd
import pickle



class Predict():
    def __init__(self):
       
        with open('conv_lstm.pkl','rb') as f:
            self.model = pickle.load(f)
        self.graph = tf.get_default_graph()
    def get_graph(self):
        return self.graph
    def intialize(self,raw_values):
        instances = ['c1.medium', 'c1.xlarge', 'c3.2xlarge', 'c3.4xlarge', 'c3.8xlarge',
               'c3.large', 'c3.xlarge', 'c4.2xlarge', 'c4.4xlarge', 'c4.8xlarge',
               'c4.large', 'c4.xlarge', 'cc2.8xlarge', 'cr1.8xlarge',
               'd2.2xlarge', 'd2.4xlarge', 'd2.8xlarge', 'd2.xlarge',
               'g2.2xlarge', 'g2.8xlarge', 'hi1.4xlarge', 'i2.2xlarge',
               'i2.4xlarge', 'i2.8xlarge', 'i2.xlarge', 'i3.16xlarge',
               'i3.2xlarge', 'i3.4xlarge', 'i3.8xlarge', 'i3.large', 'i3.xlarge',
               'm1.large', 'm1.medium', 'm1.small', 'm1.xlarge', 'm2.2xlarge',
               'm2.4xlarge', 'm2.xlarge', 'm3.2xlarge', 'm3.large', 'm3.medium',
               'm3.xlarge', 'm4.10xlarge', 'm4.16xlarge', 'm4.2xlarge',
               'm4.4xlarge', 'm4.large', 'm4.xlarge', 'r3.2xlarge', 'r3.4xlarge',
               'r3.8xlarge', 'r3.large', 'r3.xlarge', 'r4.16xlarge', 'r4.2xlarge',
               'r4.4xlarge', 'r4.8xlarge', 'r4.large', 'r4.xlarge', 't1.micro',
               'x1.16xlarge', 'x1.32xlarge']
        os = ['Linux/UNIX', 'SUSE Linux', 'Windows']
        region = ['ap-northeast-1a', 'ap-northeast-1c']

        data_columns=['OS', 'Region', 'Month', 'Week', 'Day',
               'Dayofweek', 'Dayofyear', 'Instance_Type_c1.medium',
               'Instance_Type_c1.xlarge', 'Instance_Type_c3.2xlarge',
               'Instance_Type_c3.4xlarge', 'Instance_Type_c3.8xlarge',
               'Instance_Type_c3.large', 'Instance_Type_c3.xlarge',
               'Instance_Type_c4.2xlarge', 'Instance_Type_c4.4xlarge',
               'Instance_Type_c4.8xlarge', 'Instance_Type_c4.large',
               'Instance_Type_c4.xlarge', 'Instance_Type_cc2.8xlarge',
               'Instance_Type_cr1.8xlarge', 'Instance_Type_d2.2xlarge',
               'Instance_Type_d2.4xlarge', 'Instance_Type_d2.8xlarge',
               'Instance_Type_d2.xlarge', 'Instance_Type_g2.2xlarge',
               'Instance_Type_g2.8xlarge', 'Instance_Type_hi1.4xlarge',
               'Instance_Type_i2.2xlarge', 'Instance_Type_i2.4xlarge',
               'Instance_Type_i2.8xlarge', 'Instance_Type_i2.xlarge',
               'Instance_Type_i3.16xlarge', 'Instance_Type_i3.2xlarge',
               'Instance_Type_i3.4xlarge', 'Instance_Type_i3.8xlarge',
               'Instance_Type_i3.large', 'Instance_Type_i3.xlarge',
               'Instance_Type_m1.large', 'Instance_Type_m1.medium',
               'Instance_Type_m1.small', 'Instance_Type_m1.xlarge',
               'Instance_Type_m2.2xlarge', 'Instance_Type_m2.4xlarge',
               'Instance_Type_m2.xlarge', 'Instance_Type_m3.2xlarge',
               'Instance_Type_m3.large', 'Instance_Type_m3.medium',
               'Instance_Type_m3.xlarge', 'Instance_Type_m4.10xlarge',
               'Instance_Type_m4.16xlarge', 'Instance_Type_m4.2xlarge',
               'Instance_Type_m4.4xlarge', 'Instance_Type_m4.large',
               'Instance_Type_m4.xlarge', 'Instance_Type_r3.2xlarge',
               'Instance_Type_r3.4xlarge', 'Instance_Type_r3.8xlarge',
               'Instance_Type_r3.large', 'Instance_Type_r3.xlarge',
               'Instance_Type_r4.16xlarge', 'Instance_Type_r4.2xlarge',
               'Instance_Type_r4.4xlarge', 'Instance_Type_r4.8xlarge',
               'Instance_Type_r4.large', 'Instance_Type_r4.xlarge',
               'Instance_Type_t1.micro', 'Instance_Type_x1.16xlarge',
               'Instance_Type_x1.32xlarge']
        self.final = list(np.zeros([70]))
        self.final[data_columns.index('OS')] = os.index(raw_values[0])
        self.final[data_columns.index('Region')] = region.index(raw_values[1])
        self.final[data_columns.index('Month')] = int(raw_values[2])
        self.final[data_columns.index('Week')] = int(raw_values[3])
        self.final[data_columns.index('Day')] = int(raw_values[4])
        self.final[data_columns.index('Dayofweek')] = int(raw_values[5])
        self.final[data_columns.index('Dayofyear')] = int(raw_values[6])
        instance_type = 'Instance_Type_'+raw_values[7]
        self.final[data_columns.index(instance_type)] = 1.0
        self.values =  np.array(self.final).reshape(1,1,-1,1)
        
    def pred(self):
        
        pred = self.model.predict(self.values)
        return pred
        


