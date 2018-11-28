import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
import numpy as np
# from tensorflow.contrib.legacy_seq2seq.python.ops import seq2seq
from tensorflow.contrib.rnn.python.ops import rnn
# from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl as rnn_cell  #omit when tf = 1.3
from tensorflow.python.ops import rnn_cell_impl as rnn_cell #add when tf = 1.3
# import attention_encoder
from utils import *
import Generate_stock_data as GD
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Disable Tensorflow debugging message
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

import scipy.stats
import math
#import keras
#import keras.backend as K
import os
# os.environ['KERAS_BACKEND'] = 'tensorflow'

def rrse_(y_true, y_pred):
  return np.sqrt(np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))

def CORR(y_true, y_pred):
  N = y_true.shape[0]
  total = 0.0
  for i in range(N):
    if math.isnan(scipy.stats.pearsonr(y_true[i], y_pred[i])[0]):
      N -= 1
    else:
      total += scipy.stats.pearsonr(y_true[i], y_pred[i])[0]
  return total / N

def RNN(encoder_input, decoder_input, weights, biases, encoder_attention_states, 
        n_input_encoder, n_steps_encoder, n_hidden_encoder,
        n_input_decoder, n_steps_decoder, n_hidden_decoder):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Prepare data for encoder
    # Permuting batch_size and n_steps
    encoder_input = tf.transpose(encoder_input, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    encoder_input = tf.reshape(encoder_input, [-1, n_input_encoder])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    encoder_input = tf.split(encoder_input, n_steps_encoder, 0)

    # Prepare data for decoder
    # Permuting batch_size and n_steps
    decoder_input = tf.transpose(decoder_input, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    decoder_input = tf.reshape(decoder_input, [-1, n_input_decoder])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    decoder_input = tf.split(decoder_input, n_steps_decoder,0 )

    # Encoder.
    with tf.variable_scope('encoder') as scope:
        encoder_cell = rnn_cell.BasicLSTMCell(n_hidden_encoder, forget_bias=1.0)
        encoder_outputs, encoder_state, attn_weights = attention_encoder(encoder_input,
                                         encoder_attention_states, encoder_cell)

    # First calculate a concatenation of encoder outputs to put attention on.
    top_states = [tf.reshape(e, [-1, 1, encoder_cell.output_size]) for e in encoder_outputs]
    attention_states = tf.concat(top_states,1)

    with tf.variable_scope('decoder') as scope:
        decoder_cell = rnn_cell.BasicLSTMCell(n_hidden_decoder, forget_bias=1.0)
        outputs, states, attn_weights = attention_decoder(decoder_input, encoder_state,
                                            attention_states, decoder_cell)

    return tf.matmul(outputs[-1], weights['out1']) + biases['out1'], attn_weights

def mean_absolute_percentage_error(y_true, y_pred): 
    """
    Use of this metric is not recommended; for illustration only. 
    See other regression metrics on sklearn docs:
      http://scikit-learn.org/stable/modules/classes.html#regression-metrics
    Use like any other metric
    >>> y_true = [3, -0.5, 2, 7]; y_pred = [2.5, -0.3, 2, 8]
    >>> mean_absolute_percentage_error(y_true, y_pred)
    Out[]: 24.791666666666668
    """

    # y_true, y_pred = check_arrays(y_true, y_pred)

    ## Note: does not handle mix 1d representation
    #if _is_1d(y_true): 
    #    y_true, y_pred = _check_1d_array(y_true, y_pred)

    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def root_relative_squared_error(y_true, y_pred):
    mn = np.mean(y_true)
    return np.sqrt(np.average((y_true-y_pred)**2))/np.sqrt(np.average((y_true - mn)**2))
    

def go(dataset, horizon):
    
    learning_rate = 0.001
    training_iters = 1000000
    batch_size  = 128

    model_path = './{}_model/'.format(dataset)
    filename = './data/{}/{}.txt'.format(dataset, dataset)
    df = pd.read_csv(filename,header=None)
    display_step = int(df.shape[0]*.8)//batch_size
    
    timestep = 10
    # Network Parameters
    # encoder parameter
    num_feature =  df.shape[1]-1 # number of index  #98 #72
    n_input_encoder =  num_feature # n_feature of encoder input  #98 #72
    n_steps_encoder = timestep# time steps 
    # n_hidden_encoder = 256 # size of hidden units 
    n_hidden_encoder = 64
    
    # decoder parameter
    n_input_decoder = 1
    n_steps_decoder = timestep-1
    # n_hidden_decoder = 256 
    n_hidden_decoder = 64
    n_classes = 1 # size of the decoder output

    ret_maes = []
    ret_rmses = []
    ret_mapes = []

    all_y_test = []
    all_y_pred = []
    for i in range(num_feature):
        print('predicting {} series out of {}'.format(i, num_feature))

        tf.reset_default_graph()
        # Parameters
        
        # tf Graph input
        encoder_input = tf.placeholder("float", [None, n_steps_encoder, n_input_encoder])
        decoder_input = tf.placeholder("float", [None, n_steps_decoder, n_input_decoder])
        decoder_gt = tf.placeholder("float", [None, n_classes])
        encoder_attention_states = tf.placeholder("float", [None, n_input_encoder, n_steps_encoder])
        # Define weights
        weights = {'out1': tf.Variable(tf.random_normal([n_hidden_decoder, n_classes]))}
        biases = {'out1': tf.Variable(tf.random_normal([n_classes]))}
        
        pred, attn_weights = RNN(encoder_input, decoder_input, weights, biases, encoder_attention_states,
                                 n_input_encoder, n_steps_encoder, n_hidden_encoder,
                                 n_input_decoder, n_steps_decoder, n_hidden_decoder)
        
        # Define loss and optimizer
        cost = tf.reduce_sum(tf.pow(tf.subtract(pred, decoder_gt), 2))
        loss = tf.pow(tf.subtract(pred, decoder_gt), 2)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
        init = tf.global_variables_initializer()
        
        # save the model
        saver = tf.train.Saver()
        loss_value = []
        step_value = []
        loss_test=[]
        loss_val = []

        maes = []
        rmses = []
        mapes = []
        
        
        # Launch the graph
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            
            sess.run(init)
            step = 1
            count = 1
            epochs = 50
        
            Data = GD.Input_data(batch_size, n_steps_encoder, n_steps_decoder, n_hidden_encoder, i, filename, n_classes, horizon)

            mn_validation_loss = 1e15
            ret_y_pred = ret_y_test = None
            # Keep training until reach max iterations
            while step  < training_iters:
                # the shape of batch_x is (batch_size, n_steps, n_input)
                batch_x, batch_y, prev_y, encoder_states = Data.next_batch()
                feed_dict = {encoder_input: batch_x, decoder_gt: batch_y, decoder_input: prev_y,
                            encoder_attention_states:encoder_states}
                # Run optimization op (backprop)
                sess.run(optimizer, feed_dict)
                # display the result
                if step % display_step == 0:
                    # Calculate batch loss

                    if step // display_step > epochs:
                        break

                    loss = sess.run(cost, feed_dict)/batch_size
                    print ("Epoch", step //display_step, ", Minibatch Loss= " + "{:.6f}".format(loss))
        
                    #store the value
                    loss_value.append(loss)
                    step_value.append(step)

                    # Val
                    val_x, val_y, val_prev_y, encoder_states_val = Data.validation()
                    feed_dict = {encoder_input: val_x, decoder_gt: val_y, decoder_input: val_prev_y,
                                encoder_attention_states:encoder_states_val}
                    loss_val1 = sess.run(cost, feed_dict)/len(val_y)
                    loss_val.append(loss_val1)
                    # print "validation loss:", loss_val1

                    # testing
                    test_x, test_y, test_prev_y, encoder_states_test= Data.testing()
                    # print(test_x.shape)
                    # print(test_y.shape)
                    feed_dict = {encoder_input: test_x, decoder_gt: test_y, decoder_input: test_prev_y,
                                encoder_attention_states:encoder_states_test}
                    pred_y=sess.run(pred, feed_dict)
                    loss_test1 = sess.run(cost, feed_dict)/len(test_y)
                    loss_test.append(loss_test1)
                    # print "Testing loss:", loss_test1
        
                    #save the parameters
                    # if loss_val1<=min(loss_val):
                        # save_path = saver.save(sess, model_path  + 'dual_stage_' + str(step) + '.ckpt')
        
                    mean, stdev = Data.returnMean()
                    # print mean
                    # print stdev
        
                    testing_result = test_y*stdev[num_feature] + mean[num_feature]
                    pred_result = pred_y*stdev[num_feature] + mean[num_feature]
                    
        
                    # print "testing data:"
                    # print testing_result
                    # print testing_result.shape
        
                    # print "pred data:"
                    # print pred_result
                    # print pred_result.shape
                    # from sklearn.utils import check_arrays
        
                    if loss_val1 < mn_validation_loss:

                        mn_validation_loss = loss_val1

                        ret_y_pred = pred_result.copy()
                        ret_y_test = testing_result.copy()

                        mae = mean_absolute_error(testing_result, pred_result)
                        print('mae', mae)
                        maes.append(mae)

                        rmse = np.sqrt(mean_squared_error(testing_result, pred_result))
                        # print('rmse', rmse)
                        rmses.append(rmse)

                        mape = mean_absolute_percentage_error(testing_result, pred_result)
                        # print('mape', mape)
                        mapes.append(mape)

        
                step += 1
                count += 1
        
                # reduce the learning rate
                if count > 10000:
                    learning_rate *= 0.1
                    count = 0
                    # save_path = saver.save(sess, model_path  + 'dual_stage_' + str(step) + '.ckpt')
        
            print ("Optimization Finished!")
            all_y_pred.append(ret_y_pred.flatten())
            all_y_test.append(ret_y_test.flatten())
            print(np.array(all_y_pred).shape)
            print(np.array(all_y_test).shape)

            rrse = rrse_(np.array(all_y_test), np.array(all_y_pred))
            corr = CORR(np.array(all_y_test) ,np.array(all_y_pred))
            print('current score', rrse, corr)

            # ret_maes.append(min(maes))
            # ret_rmses.append(min(rmses))
            # ret_mapes.append(min(mapes))
            # print(all_y_pred)
            # input()
            # print(all_y_test)
            # input()
    
    # print(ret_maes, ret_rmses, ret_mapes)
    # return np.mean(ret_maes), np.mean(ret_rmses), np.mean(ret_mapes)
    # df = pd.DataFrame(all_pred_val, columns=["pred_val"])
    # df.insert(loc=1, column='test_val', value=all_test_val)
    # df.to_csv('./result/nas.csv', index=False) #/US10YT=RR_30days
    return np.array(all_y_pred), np.array(all_y_test)

if __name__ == '__main__':
    # datasets = ['electricity', 'exchange_rate', 'solar-energy', 'traffic']
    # datasets = datasets[1:]
    # datasets = datasets[::-1]
    import sys
    datasets = [sys.argv[1]]
    horizons = [int(sys.argv[2])]
    print(datasets, horizons)
    #horizons = [3, 6, 12, 24]

    f = open('log', 'a+')
    f.write('dataset,horizon,mae,rmse,mape\n')
    for dataset in datasets:
        for horizon in horizons:
            print(dataset, horizons)
            y_pred, y_test = go(dataset, horizon)
            print(y_pred.shape, y_test.shape)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mape = mean_absolute_percentage_error(y_test, y_pred)
            rrse = rrse_(y_test, y_pred)
            corr = CORR(y_test ,y_pred)
            # mae, rmse, mape = go(dataset, horizon)
            f.write('{},{},{},{},{},{},{}\n'.format(dataset, horizon, mae, rmse, mape, rrse, corr))
            f.flush()

    f.close()
