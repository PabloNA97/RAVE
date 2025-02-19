"""RAVE: using predicitve inforamtion bottleneck framework to learn RCs 
to enhance the sampling of MD simulation. Code maintained by Yihang.

Read and cite the following when using this method:
https://aip.scitation.org/doi/abs/10.1063/1.5025487
https://www.nature.com/articles/s41467-019-11405-4
https://arxiv.org/abs/2002.06099
"""
import json
import numpy as np
import COLVAR2npy
import Analyze_prave
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, Lambda
from keras.initializers import RandomUniform, Constant
from keras.optimizers import RMSprop
from keras.constraints import unit_norm
from keras import regularizers
from keras.callbacks import Callback
from keras.losses import mean_squared_error
from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()

########################
### Global Functions ###
def data_prep(system_name, number_trajs, predictive_step):
    """ Read the input trajectory files.
        Prepare x, x_t trajectory and corresponding re-weighting factors
        
        Parameters
        ----------
        system_name : string
            Name of the system.
            
        number_trajs : int
            Number of trajectories.
        
        predictive_step : int
            Predictive time delay.
        
        Returns
        -------
        X : np.array
            present trajectory.
         
        Y : np.array
            future trajectory.
            
        W1 : np.array
            re-weighting factors in objective function before P(X_t | \chi )
            
        W2 : np.array
            re-weighting factors in objective function before P(X | \chi )
    """
    
    
    for j in range(number_trajs):

        traj_file_name = 'input/x_'+system_name+'_%i.npy'%j   # present trajectory of the shape n*d, where n is the MD steps and d is the number of order parameters (PLUMED output)
        w_file_name = 'input/w_'+system_name+'_%i.npy'%j      # weights correspond to trajectory in x. Calculated by exp(beta*V)   

        # If time delay = 0
        if predictive_step==0:

            x = np.load(traj_file_name)
            y = x[:,:]      

            w1 =  np.load(w_file_name)   
            w2 = np.zeros( np.shape(w1) )      

        # If time delay =! 0
        else:

            x = np.load(traj_file_name)

            # Save data, shifted "time delay" wrt each other
            y = x[predictive_step: , :]  # X_dt (advanced dt with respect to X)
            x = x[:-predictive_step, :]  # X

            w = np.load(w_file_name)

            # Save corresponding weights
            w_x = w[:-predictive_step] # weights for X
            w_y = w[predictive_step:]  # weights for X_dt   

            # Will be used in the loss function? 
            w1 = ( w_x * w_y )**0.5
            w2 =  w_x**0.5*( w_x**0.5 - w_y**0.5)

        try:
            X = np.append(X, x, axis = 0)
            Y = np.append(Y, y, axis = 0)

            W1 = np.append(W1, w1, axis = 0)
            W2 = np.append(W2, w2, axis = 0)

        except:
            X = x
            Y = y

            W1 = w1
            W2 = w2

    normalization_factor = np.sum(W1)/len(W1)  

    W1 /= normalization_factor
    W2 /= normalization_factor    

    print('length of data: %i'%np.shape(X)[0] )
    print('number of order parameters: %i'%np.shape(X)[1] )
    print('min re-weighting factor: %f'%np.min(W1))
    print('max re-weighting factor: %f'%np.max(W1))   

    return X, Y, W1, W2

def random_pick(x, x_dt, w1, w2, training_len):
    """ ramdomly pick (x, x_dt) pair from data set of size "training_length"
    
        Parameters
        ----------
        x : np.array
            present trajectory.
         
        x_dt : np.array
            future trajectory.
            
        w1 : np.array
            re-weighting factors in objective function before P(X_t | \chi )
            
        w2 : np.array
            re-weighting factors in objective function before P(X | \chi )
            
        training_len: int
            length of the return data set
            
        
        Returns
        -------
        x1 : np.array
            randomly selected data points from present trajectory.
         
        x2 : np.array
            future trajectory corresponds to selected data points in x1.
            
        w1 : np.array
            corresponding re-weighting factors in objective function before P(X_t | \chi )
            
        w1 : np.array
            corresponding re-weighting factors in objective function before P(X | \chi )
    """   
    
    indices = np.arange( np.shape(x)[0])   
    np.random.shuffle(indices)
    indices = indices[:training_len]
    x = x[indices, :]
    x_dt = x_dt[indices, :]
    w1 = w1[indices]
    w2 = w2[indices]
    print('%i data points are used in this training'%len(indices))
    
    return x, x_dt, w1, w2

def scaling(x):
    """ make order parameters with mean 0 and variance 1
        return new order parameter and scaling factors
        
        Parameters
        ----------
        x : np.array
            order parameters
            
        Returns
        ----------
        x : np.array
            order parameters after rescaling
        
        std_x : np.array
            rescaling factors of each OPs
              
     """ 
   
    x = x-np.mean(x, axis =0)
    std_x = np.std(x, axis =0)
    return x/std_x, std_x

def sampling(args):
	"""Sample the latent variable
	from a Normal distribution."""
	s_mean= args
	epsilon = K.random_normal(shape=(batch_size,rc_dim), mean=0.0, stddev=s_vari )
	s_noise = s_mean +  epsilon
	return s_noise


def dynamic_correction_loss(x, w1, w2):
    """Custom loss function with dynamic correction
       
       Parameters:
       -----------
       x: input data
       w1:
       w2:
       """

    def custom_loss(y_true, y_pred ):
         ce1 = mean_squared_error(y_true, y_pred )
         ce2 = mean_squared_error(x, y_pred)  
         return (w1[:,0]*ce1+w2[:,0]*ce2) 

    return custom_loss

class WeightsHistory(Callback):

    def on_train_begin(self, logs={}):
        self.losses = []
        self.losses_vali = []
        self.weights0 = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.losses_vali.append(logs.get('val_loss'))
        self.weights0.append( prave.layers[1].get_weights()) # Coefficients of first layer
    

#########################

if __name__ == '__main__':
    
    ### Global Variables ###
    #system info   
    system_name = 'roha'
    n_trajs = 1                  # number of trajectories
    save_path = 'output/'        # path to the directory that saves output files
    T = 300                      # Temperature in unit of Kelvin 
    bias = False                 # When false reweigting factors are set to 1. 
                                 # When true, reweigting factors are calculated and save 
    
    # Predictive time delay ### (2 ps)
    time_delay= [750, 1500, 2500, 3000, 3500] # list(range(0, 100, 10)) #predictive time delay
    
    # Network variables
    training_size = 2500000  # if training_size = n, only n data points will be randomly piked from the whole data set and used to do the training
    batch_size = 5000        # total number of training data point n should be a multiple of batch_size 
    op_dim = 10               # dimensionality of order parameters
    rc_dim = 1                # dimensionality of reaction coordinates 
    int_dim = 128             # number of cells in each layer
    s_vari = 0.005 
    learning_rate = 0.0002 
    decay = 0.0       
    trials = range(1) 
    epochs = 100              # Number of epochs to train the model
    random_uniform = RandomUniform(minval=-0.5, maxval=0.5) 
    set_constant = Constant(value = 0.5**0.5)
    if_whiten = True         # If whiten then normalize the order parameters (mean 0 and variance = 1)
    
    # Convert COLVAR file to npy file - for each traj
    for traj_index in range(n_trajs):

        input_name = system_name+'_%i'%traj_index
        input_dir = 'input/'

        COLVAR2npy.COLVAR2npy( input_name, T, op_dim, input_dir, bias )
      
    ### set predictive time delay ###
    if not bias:
        system_name = 'unbiased_' + system_name       
   
    ######################## 
    for dt in time_delay:      
        ######################## 
        ### load the dataset ###
        (x, y, w1, w2) = data_prep( system_name, n_trajs, dt )

        if if_whiten:
            x, scaling_factors = scaling(x) 
            y -= np.mean( y, axis =0)
            y /= scaling_factors
        else:
            scaling_factors = np.ones( op_dim )

        ############################   
        ### run different trials ###  # trials for training
        for trial in trials:   

            Result = []

            ############################################  
            ### Variational Autoencoder architecture ###
            input_Data = Input(batch_shape=(batch_size, op_dim))  
            input_w1 = Input(shape=(1,))    
            input_w2 = Input(shape=(1,))  
            linear_encoder = Dense( rc_dim, activation=None, use_bias=None,  kernel_regularizer=regularizers.l1(0.0), kernel_initializer='random_uniform',  kernel_constraint = unit_norm(axis=0))(input_Data)
                   
            s = Lambda(sampling)(linear_encoder)  # Custom layer with the gaussian sampling
            hidden_a = Dense(int_dim, activation='elu', kernel_initializer='random_uniform')(s)
            hidden_b = Dense(int_dim, activation='elu', kernel_initializer='random_uniform')(hidden_a)
            y_reconstruction = Dense( op_dim, activation=None, kernel_initializer='random_uniform')(hidden_b)
            
            ######################################### 
            ### Randomly pick samples from dataset ###
            # Is it ok the overlap between training and validation set?
            # Maybe we should choose training size and validation size << total size
            
            # data for training
            train_x, train_y, train_w1, train_w2 = random_pick(x, y, w1, w2,training_size)
            # data for validation
            vali_x, vali_y, vali_w1, vali_w2 = random_pick(x , y, w1, w2, training_size) 
           
            #############################################
            ### Prepare the PRAVE and train the PRVAE ###

            # Put the model togather
            prave = Model([input_Data, input_w1 , input_w2] ,y_reconstruction)            
            
            # Create optimizer class
            rmsprop = RMSprop(learning_rate=learning_rate, decay = decay)
            
            # Compile the model
            prave.compile(optimizer=rmsprop,loss=dynamic_correction_loss(input_Data, input_w1, input_w2))
            
            # Create class "WeightsHistory"
            history = WeightsHistory()

            # Fit the model, save loss and validation loss on epoch end
            History = prave.fit( [train_x,train_w1,train_w2], train_y,
        	    shuffle=True,
        	    epochs=epochs,
        	    batch_size=batch_size,
                validation_data=([vali_x,vali_w1,vali_w2], vali_y),
                callbacks = [history] )
                    
            ####################
            ### Save results ###
            Loss = np.array(  history.losses  )
            Val_Loss = np.array(  history.losses_vali  )     

            # Coefficients of first layer (linear encoder)    
            Weights0=np.array( history.weights0 )[:,0,:,:]  #  num epochs x ? x dim OP x dim latent space (num RC)
            
            # w_norm = np.linalg.norm(Weights0,  axis=1)
            for op_index in range( op_dim ):
                Weights0[:,op_index,:] /= scaling_factors[op_index] # rescale back to RC weights of non-standardized OPs

            for rc_index in range( rc_dim ):
                Weights0[:, :, rc_index] = np.transpose( np.transpose( Weights0[:, :, rc_index] ) / np.linalg.norm(Weights0[:, :, rc_index], axis=1)) #normalize the rc weights
                 
            Loss = np.expand_dims(Loss, axis=-1)
            Val_Loss = np.expand_dims(Val_Loss, axis=-1)

            result_loss = np.concatenate((Loss, Val_Loss) , axis =-1)

            result_weights = Weights0         

            K.clear_session()

            print('!!!!')
            print(np.shape(result_weights))
            
            # Name for results - to avoid overwriting different dt and trials for a certain set of hyper-parameters
            save_info = system_name+'_dt'+str(dt)+'_trial'+str(trial) 

            # Save loss and coefficients
            np.save(save_path+'Loss_'+save_info, result_loss)
            np.save(save_path+'Weights_'+save_info, result_weights)
    
    # Hyper-parameters 
    hyper_param ={  
        "train_size": training_size,
        "batch_size": batch_size,
        "epochs": epochs,
        "trials": trial+1,
        "layer_dimension": int_dim,
        "learning_rate": learning_rate,
        "decay":decay,
        "decoder_noise_variance": s_vari
    } 
      
    # Dump hyperparameters  
    json_object = json.dumps(hyper_param, indent = 4)

    # Writing to sample.json
    json_file = save_path + "hyper_parameters.json"
    with open(json_file, "w") as outfile:
        outfile.write(json_object)

    ### analyze and save the results ###
    Analyze_prave.save_result(system_name, op_dim, time_delay, trials, save_path)  


        

