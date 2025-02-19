"""Data preparation part of RAVE.
Code maintained by Yihang.

Read and cite the following when using this method:
https://aip.scitation.org/doi/abs/10.1063/1.5025487
https://www.nature.com/articles/s41467-019-11405-4
https://arxiv.org/abs/2002.06099
"""
import numpy as np 
import os.path
from os import path

def COLVAR2npy( Name, Temperature, OP_dim, Dir ='input/', Bias = False ):
    ''' Covert the COLVAR (file that cotains the trajectory and corresponding biasing potential)
        to npy file.
        Here we assume that the COLVAR is of the shape n*(1+d1+d2+d3)
        n is the number of data points printed by plumed
        the first column is the simulation time, usually in unit of ps
        d1 is the dimensionality of order parameters
        d2 is the dimensionality of reaction coordinates
        d3 is the number of bias potentials that is added during the simulation
        
        Parameters
        ----------
        Name : string
            Name of the system.
            
        Temperature : float
            Temperature in unit of Kelvin.
        
        OP_dim : int
            Dimensionality of order parameter space (feature space).
        
        Dir: string
            Directory of input and output (.npy) files.
        
        Bias: 
            Whether the trajectory is from a biased MD.
            When false re-weighting factors are set to 1. 
            When true, re-weighting factors are calculated and save. 
    
        Returns
        -------
        None
        
    '''  
    n_bias_convert = 1  # number of biases that will be converted into re-weighting factors = total number of biases - biases from linear CV already learned
    t0 = 0              # initial MD step
    total_steps = -1    # total number of MD step, so only the date from t0 to t0+total_steps will be saved 

    # If the traj is biased 
    if Bias:

        # Files to save data (x) and weights (w)
        traj_save_name = Dir + 'x_'+Name
        bias_save_name = Dir + 'w_'+Name

        if path.exists(traj_save_name + '.npy') and path.exists(bias_save_name + '.npy'):
            print( 'npy files already exist, delete them if you want to generate new ones.')
        else:

            # Load input data
            Data = np.loadtxt(Dir+ Name)   

            # x: time and the value of the order parameters for all frames
            x = Data[::, 1:OP_dim +1]  # all rows, columns of OP

            # w has the value of the bias for every frame - associated to a certain value of the RC and order parameters
            w = np.sum( Data[::, -n_bias_convert :], axis = 1)  # All rows, summing all columns corresponding to biases we want to re-weight

            # only a part of the full trajectory will be saved
            x = x[t0:t0+total_steps, :]   # rows corresponding to "t0" to "t0 + total_steps" (time steps), all columns (OP) 
            w = w[t0:t0+total_steps]      # Total bias to re-weight at each time step / sample
            
            # Save OPs
            np.save(Dir+'x_'+ Name, x)
            
            # Calculate re-weighting factor for each sample
            re_weighting_factor = np.exp(w/(0.008319*Temperature))   # Here we assume that unit if bias is kJ
            
            # Save re-weighting factors
            np.save(Dir+'w_'+ Name, re_weighting_factor) 

    # If the traj is unbiased 
    else: 

        # Files to save data (x) and weights (w)
        traj_save_name = Dir + 'x_unbiased_' + Name
        bias_save_name = Dir + 'w_unbiased_' + Name     

        if path.exists( traj_save_name+'.npy') and path.exists( bias_save_name+'.npy'):
            print( 'npy files already exit, delete them if you want to generate new ones.')
        else:

            # Load input data
            Data = np.loadtxt(Dir+ Name) 

            # x: time and the value of the order parameters for all frames
            x = Data[::, 1:OP_dim +1]     # all rows, columns of OP

            #only a part of the full trajectory will be saved
            x = x[t0:t0+total_steps, :] # rows corresponding to "t0" to "t0 + total_steps" (time steps), all columns (OP) 

            # Save OPs
            np.save(Dir+'x_unbiased_'+ Name, x)

            # trajectories are treated as unbiased - re-weighting factors = 1
            re_weighting_factor = np.ones( np.shape(x[:,0]))     

            # Save re-weighting factors
            np.save(Dir+'w_unbiased_'+ Name, re_weighting_factor)

if __name__ == '__main__':
    
    system_name = 'roha'
    n_trajs = 1                   # number of trajectories
    save_path = 'output/'         # path to the directory that saves output files
    T = 300                       # Temperature in unit of Kelvin 
    bias = False                  # When false reweigting factors are set to 1. 
    op_dim = 10                   # dimension of order parameters

    for traj_index in range(n_trajs):
        COLVAR2npy.COLVAR2npy( system_name+'_%i'%traj_index, T, op_dim, 'input/', bias )
      
    
    
        
        
    




