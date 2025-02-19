# -*- coding: utf-8 -*-
#Author: Yihang Wang
#Data: June 28, 2019
import numpy as np


def save_result(system_name, op_dim, time_delay, trials, save_path='output/'):
    ''' save final result (linear combinaton coefficients of OPs) to a txt file
        Parameters
        ----------
        system_name : string
            Name of the system.
            
        op_dim : int
            Dimensionality of order parameters.
        
        time_delay : int
            Predictive time delay.
        
        trials: list
            Indexes of all trials.
        
        save_path: string
            Directory of where the final result is saved.
            
        Returns
        ----------
            None  
            Result is saved to a txt file.
    
    '''
    weights = []
    for dt in time_delay:
        Loss = []
        Weights = []
        for trial in trials: 
            save_dir = system_name+'_dt'+str(dt)+'_trial'+str(trial)+'.npy'
            Result_loss = np.load(save_path+'Loss_'+save_dir) 
            Result_weights = np.load(save_path+'Weights_'+save_dir) 
            Loss.append(np.average( Result_loss[-2:,-1] ))
            Weights.append( Result_weights[-1,:,:] )
        
        Weights = np.array( Weights )
        min_index = np.argmin(Loss)
        weights.append( Weights[min_index,:,:] )
    weights = np.array(weights)
    
    ###save weights vs. time delay###
    head = 'time_delay/MD_step  '
    number_rcs = np.shape(Weights)[-1]
    print('There are %i reaction coordinates'%number_rcs)
    for j in range(op_dim):
        head+='op%i  '%(j+1)
    for j in range(len(time_delay)):
         result_given_dt = np.concatenate((np.transpose( [[time_delay[j]]*number_rcs] ), np.transpose(weights[j,:,:])), axis =-1)
         try:
             final_result = np.concatenate((final_result, result_given_dt), axis=0)
         except:
             final_result = result_given_dt
            
    np.savetxt( save_path+'final_result_'+system_name+'.txt', final_result, header =head, newline='\n')
    

         
if __name__ == '__main__':    
########################
#generate txt file for bias potential
########################    
    #file info   
    system_name = 'roha '
    save_path = 'output/'        #pth to the directory that saves output files
        
    ###predictive time delay ###
    time_delay= list(range(0, 100, 10)) #predictive time delay

    #network variables
    training_size = 500000  # if training_size = n, only n data points will be randomly piked from the whole data set and used to do the training
    batch_size = 5000       # total number of training data point n should be a multiple of batch_size 
    op_dim = 14             # dimensionality of order parameters
    rc_dim = 1              # dimensionality of reaction coordinates 
    int_dim = 128           # number of cells in each layer
    s_vari = 0.005 
    learning_rate = 0.0002 
    decay = 0.0       
    trials = range(1) 

    network_info = '_int_dim'+str(int_dim)+'_lr'+str(learning_rate)+'_decay'+str(decay)+'_batch_size'+str(batch_size)
    save_result(system_name, op_dim, time_delay, trials, s_vari, training_size, network_info, save_path)  
        
    
   
