import os
import numpy as np
from keras.models import Sequential
from keras.layers import Conv1D,BatchNormalization
from keras.layers import Activation,Dropout,Dense,Flatten,MaxPooling1D
from keras.layers import LSTM,Bidirectional,TimeDistributed,RepeatVector
import pandas as pd
import pickle
import matplotlib.pyplot as plt


def data_gen(path, batchsize,features,fun='CNN',steps=1):
    inputs = []
    targets = []
    files = os.listdir(path)
    batchcount = 1
    while True:
        with open(path+'/'+files[0]) as data,open(path+'/'+files[1]) as output:
            for line,op in zip(data,output):
                x = line.split(',')
                inputs.append(x)
                targets.append(op)
                
                batchcount += 1
                
                if batchcount > batchsize:
                    
                    X = np.array(inputs, dtype='float32')
                    y = np.array(targets, dtype='float32')
                    if fun == 'ConvLSTM':
                        X = X.reshape((X.shape[0], 1, features,steps))
                        #y = y.reshape((y.shape[0],1,1))
                    else :
                        X = X.reshape(X.shape[0],X.shape[1],1)
                    yield (X, y)
                    inputs = []
                    targets = []
                    batchcount = 1


# In[61]:


class CNN():
    def __init__(self,data_train,data_val,epochs=20,verbose=1,batch_size=8):
        
        self.train_path = data_train
        self.val_path = data_val
        self.n_steps= 1
        self.n_outputs = 1
        self.epochs= epochs
        self.verbose = 1 
        self.batch_size = batch_size
        
        df = pd.read_csv(data_train+'/train.csv',header=None)
        self.n_features = df.shape[1]
        self.num_rows = df.shape[0]
        del df
        df = pd.read_csv(data_val+'/test_op.csv',header=None)
        self.validation_steps =df.shape[0]
        del df
        
        # define model

        self.model = Sequential(name = 'CNN')
        
        #1st Convolution
        self.model.add(Conv1D(filters=8, kernel_size=4, input_shape=(self.n_features,self.n_steps)))
        self.model.add(BatchNormalization())
        self.model.add(Activation('relu'))
        
        #Neural Architechture
        self.model.add(Flatten())
        self.model.add(Dense(64,activation='relu'))
        self.model.add(Dense(self.n_outputs))
    
    def train(self):
    
        self.model.compile(loss="mse", optimizer='adam',metrics=['mse'])
        print(self.model.summary())
        # fit model
        self.history_cnn = self.model.fit(data_gen('train',self.batch_size,features=self.n_features),validation_data=data_gen('validation',self.batch_size,features=self.n_features), epochs=self.epochs,
                                     verbose=self.verbose,steps_per_epoch=self.num_rows/self.batch_size,
                                    validation_steps = self.validation_steps)
        pickle.dump(self.model,open('cnn_model.pkl', 'wb'))
    
    def Plot(self):
        mse = self.history_cnn.history['mean_squared_error']
        plt.title("Training MSE")
        plt.plot(mse,"*-")
        plt.show()
        val_mse = self.history_cnn.history['val_mean_squared_error']
        plt.title("Validation MSE")
        plt.plot(val_mse,"*-")
        plt.show()


# In[62]:


class Lstm():
    def __init__(self,data_train,data_val,epochs=20,verbose=1,batch_size=8):
        
        self.train_path = data_train
        self.val_path = data_val
        self.n_steps= 1
        self.n_outputs = 1
        self.epochs= epochs
        self.verbose = 1 
        self.batch_size = batch_size
        
        
        df = pd.read_csv(data_train+'/train.csv',header=None)
        self.num_rows = df.shape[0]
        self.n_features = df.shape[1]
        df = pd.read_csv(data_val+'/test_op.csv',header=None)
        self.validation_steps =df.shape[0]
        del df
        
        self.model = Sequential()
        self.model.add(LSTM(8, return_sequences=True,input_shape=(self.n_features,self.n_steps)))
        self.model.add(Bidirectional(LSTM(8)))
        self.model.add(Dense(self.n_outputs))
        
    
    def train(self):
        self.model.compile(loss="mse", optimizer='adam',metrics=['mse'])
        print(self.model.summary())
        self.history_lstm = self.model.fit(data_gen(self.train_path,self.batch_size,features=self.n_features),validation_data=data_gen(self.val_path,self.batch_size,features=self.n_features), epochs=self.epochs,
                                     verbose=self.verbose,steps_per_epoch=self.num_rows/self.batch_size,
                                    validation_steps = self.validation_steps)
        pickle.dump(self.model,open('lstm_model.pkl', 'wb'))

    def Plot(self):
        mse = self.history_lstm.history['mean_squared_error']
        plt.title("Training MSE")
        plt.plot(mse,"*-")
        plt.show()
        val_mse = self.history_lstm.history['val_mean_squared_error']
        plt.title("Validation MSE")
        plt.plot(val_mse,"*-")
        plt.show()
# In[11]:


class ConvLstm():
    def __init__(self,data_train,data_val,epochs=20,verbose=1,batch_size=8):
        
        self.train_path = data_train
        self.val_path = data_val
        self.n_steps= 1
        self.n_outputs = 1
        self.epochs= epochs
        self.verbose = 1 
        self.batch_size = batch_size
        
        
        df = pd.read_csv(data_train+'/train.csv',header=None)
        self.num_rows = df.shape[0]
        self.n_features = df.shape[1]
        df = pd.read_csv(data_val+'/test_op.csv',header=None)
        self.validation_steps =df.shape[0]
        del df
        
        self.model = Sequential()
        self.model.add(TimeDistributed(Conv1D(filters=128, kernel_size=1, activation='relu'), input_shape=(None,self.n_features,self.n_steps)))
        self.model.add(TimeDistributed(MaxPooling1D(pool_size=1)))
        self.model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu')))
        self.model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu')))
        self.model.add(TimeDistributed(Flatten()))
        self.model.add(LSTM(250, activation='relu', return_sequences=True))
        self.model.add(LSTM(150, activation='relu', return_sequences=True,dropout=0.2))
        self.model.add(LSTM(128, activation='relu',return_sequences=True))
        self.model.add(Bidirectional(LSTM(128, activation='relu')))
        self.model.add(Dense(self.n_outputs))
   
    
    def train(self):
        self.model.compile(loss="mse", optimizer='adam',metrics=['mse'])
        print(self.model.summary())
        self.history_Convlstm = self.model.fit(data_gen(self.train_path,self.batch_size,fun="ConvLSTM",features=self.n_features),validation_data=data_gen(self.val_path,self.batch_size,fun="ConvLSTM",features=self.n_features), epochs=self.epochs,
                                     verbose=self.verbose,steps_per_epoch=self.num_rows/self.batch_size,
                                    validation_steps = self.validation_steps)
        pickle.dump(self.model,open('conv_lstm.pkl', 'wb'))
    
    def Plot(self):
        mse = self.history_Convlstm.history['mean_squared_error']
        plt.title("Training MSE")
        plt.plot(mse,"*-")
        plt.show()
        val_mse = self.history_Convlstm.history['val_mean_squared_error']
        plt.title("Validation MSE")
        plt.plot(val_mse,"*-")
        plt.show()
        
