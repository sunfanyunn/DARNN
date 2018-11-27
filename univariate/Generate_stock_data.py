import numpy as np
import random
import pandas as pd
class Input_data:
    def __init__(self, batch_size, n_step_encoder, n_step_decoder, n_hidden_encoder, filename, n_label=None, horizon=3):                                   
        self.horizon = horizon
        # read the data 
        data = pd.read_csv(filename)
        if 'key' in data.columns:
            data.drop(columns=['key'], inplace=True)
        if 'pm2.5' in data.columns:
            data = data.iloc[24:, :]
            data = data.interpolate()
            # data.fillna(0, inplace=True)
        else:
            # default 
            print('fillna 0')
            data.fillna(0, inplace=True)

    
        self.data = np.array(data)
        # self.train_day =  300 # stat from 62 ~ 70     
        # self.val_day = 16
        # self.test_day = 16
        # minutes = 1

        sz = self.data.shape[0]
        train_size = int(sz * .7)
        val_size = int(sz * .8)

        self.train = self.data[0:train_size, :]
        self.val = self.data[train_size:val_size, :]
        self.test = self.data[val_size:, :]
        
        # parameters for the network                 
        self.batch_size = batch_size
        self.n_hidden_state = n_hidden_encoder
        self.n_step_encoder = n_step_encoder
        self.n_step_decoder = n_step_decoder
        

        self.n_train = len(self.train)
        self.n_val = len(self.val)
        self.n_test = len(self.test)
        self.n_feature = self.data.shape[1]- 1
        self.n_label = n_label
        
        # data normalization
        self.mean = np.mean(self.train,axis=0)
        self.stdev = np.std(self.train,axis=0)
    

        # in case the stdev=0,then we will get nan
        for i in range (len(self.stdev)):
            if self.stdev[i] < 0.00000001:
                self.stdev[i] = 1
       
    
        self.train = (self.train-self.mean)/self.stdev
        self.test = (self.test-self.mean)/self.stdev
        self.val = (self.val - self.mean)/self.stdev
        print(self.train.shape, self.val.shape, self.test.shape)

        
                         
    def next_batch(self):
        # generate of a random index from the range [0, self.n_train -self.n_step_decoder +1]                 
        if self.n_label == 1:
            index = random.sample(np.arange(0,self.n_train-self.n_step_decoder-self.horizon+1),self.batch_size)  
            # index = np.arange(0,self.n_train-self.n_step_decoder) 
            np.random.shuffle(index)
            index = np.array(index)
            # the shape of batch_x, label, previous_y                 

            # batch_x = np.zeros([index.shape[0],self.n_step_encoder, self.n_feature])
            # label = np.zeros([index.shape[0], self.n_label])
            # previous_y = np.zeros([index.shape[0],self.n_step_decoder, self.n_label])                      
            batch_x = np.zeros([self.batch_size,self.n_step_encoder, self.n_feature])
            label = np.zeros([self.batch_size, self.n_label])
            previous_y = np.zeros([self.batch_size,self.n_step_decoder, self.n_label])                      

            temp = 0
            for item in index:
                batch_x[temp,:,:] = self.train[item:item+self.n_step_encoder, :self.n_feature]             
                previous_y[temp,:,0] = self.train[item:item + self.n_step_decoder, -1]
                temp += 1
            label[:,0] = np.array(self.train[index + self.n_step_decoder + self.horizon - 1, -1])                 
            encoder_states = np.swapaxes(batch_x, 1, 2)
            return batch_x, label, previous_y, encoder_states

        # index = random.sample(np.arange(0,self.n_train-self.n_step_decoder-self.n_label),self.batch_size)  
        # index = np.array(index)
        # # the shape of batch_x, label, previous_y                 
        # batch_x = np.zeros([self.batch_size,self.n_step_encoder, self.n_feature])
        # label = np.zeros([self.batch_size, self.n_label])
        # previous_y = np.zeros([self.batch_size,self.n_step_decoder, 1])                      
        # # print(batch_x.shape, label.shape, previous_y.shape)
        # temp = 0
        # for item in index:
            # batch_x[temp,:,:] = self.train[item:item+self.n_step_encoder, :self.n_feature]             
            # previous_y[temp,:,0] = self.train[item:item + self.n_step_decoder, -1]
            # label[temp,:] = self.train[item+self.n_step_decoder: item+self.n_step_decoder + self.n_label, -1]
            # temp += 1
        # # label[:,0] = np.array(self.train[index + self.n_step_decoder, -1])                 
        # encoder_states = np.swapaxes(batch_x, 1, 2)
        # return batch_x, label, previous_y, encoder_states

    def returnMean(self):
        return self.mean, self.stdev
        
    def validation(self):  
        index = np.arange(0, self.n_val-self.n_step_decoder-self.horizon+1)
        index_size = len(index)
        val_x = np.zeros([index_size, self.n_step_encoder, self.n_feature])
        val_label = np.zeros([index_size, self.n_label])
        val_prev_y = np.zeros([index_size, self.n_step_decoder, self.n_label])
        temp = 0
        for item in index:
            val_x[temp,:,:] = self.val[item:item + self.n_step_encoder, :self.n_feature]
            val_prev_y[temp,:,0] = self.val[item:item + self.n_step_decoder, -1]        
            temp += 1

        val_label[:, 0] = np.array(self.val[index + self.n_step_decoder + self.horizon - 1, -1])
        encoder_states_val = np.swapaxes(val_x,1,2)
        return val_x, val_label, val_prev_y, encoder_states_val
        
    def testing(self):
        if self.n_label == 1:
            index = np.arange(0,self.n_test-self.n_step_decoder-self.horizon+1)
            index_size = len(index)
            test_x = np.zeros([index_size, self.n_step_encoder, self.n_feature])
            test_label = np.zeros([index_size, self.n_label])
            test_prev_y = np.zeros([index_size, self.n_step_decoder, self.n_label])
            temp = 0
            for item in index:
                test_x[temp,:,:] = self.test[item:item + self.n_step_encoder, :self.n_feature]
                test_prev_y[temp,:,0] = self.test[item:item + self.n_step_decoder, -1]        
                temp += 1

            test_label[:, 0] = np.array(self.test[index + self.n_step_decoder + self.horizon - 1, -1])
            encoder_states_test = np.swapaxes(test_x,1,2)
            return test_x, test_label, test_prev_y, encoder_states_test
        else:
            index = np.arange(0,self.n_test-self.n_step_decoder-self.n_label)
            index_size = len(index)
            test_x = np.zeros([index_size, self.n_step_encoder, self.n_feature])
            test_label = np.zeros([index_size, self.n_label])
            test_prev_y = np.zeros([index_size, self.n_step_decoder, 1])
            temp = 0
            for item in index:
                test_x[temp,:,:] = self.test[item:item + self.n_step_encoder, :self.n_feature]
                test_prev_y[temp,:,0] = self.test[item:item + self.n_step_decoder, -1]        
                test_label[temp, :] = np.array(self.test[item+ self.n_step_decoder:item+ self.n_step_decoder + self.n_label, -1])
                temp += 1

            encoder_states_test = np.swapaxes(test_x,1,2)
            return test_x, test_label, test_prev_y, encoder_states_test
