# -*- coding: utf-8 -*-

__all__ = ['TemplateLogic']


import os
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import polygamma
from scipy.stats import gamma,norm
from scipy.signal import convolve
from tifffile import imread, TiffFile
from multiprocessing import Queue, Process
from dataclasses import dataclass
import matplotlib.animation as animation


from qudi.core.module import LogicBase
from qudi.core.statusvariable import StatusVar
from qudi.core.configoption import ConfigOption


# qudi logic measurement modules must inherit qudi.core.module.LogicBase or other logic modules.
class MoleculeDetectionLogic(LogicBase):
    """ This is a simple template logic measurement module for qudi.

    Example config that goes into the config file:

    example_logic:
        module.Class: 'template_logic.TemplateLogic'
        options:
            increment_interval: 2
        connect:
            template_hardware: dummy_hardware
    """
    _gamma_var = ConfigOption(name='gamma_var', default=2.7, missing='warn')
    
    # Declare static parameters that can/must be declared in the qudi configuration
    _increment_interval = ConfigOption(name='increment_interval', default=1, missing='warn')

    # Declare status variables that are saved in the AppStatus upon deactivation of the module and
    # are initialized to the saved value again upon activation.
    _counter_value = StatusVar(name='counter_value', default=0)

    # Declare connectors to other logic modules or hardware modules to interact with
    _template_hardware = Connector(name='template_hardware',
                                   interface='TemplateInterface',
                                   optional=True)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sigma = (1/2.3548) * self._gamma_var

    def on_activate(self) -> None:
        # stuff
        pass

    def on_deactivate(self) -> None:
        # stuff
        pass
    def ln_transform(self, r_mn,theta):
        """
        log transform function use for the gamma distribution background
        Parameters :
        * `r_mn` : array given to get its log transform
        * `theta` : scale parameter from the gamma distribution
        """
        return np.log(r_mn, dtype="float16")-np.log(theta, dtype="float16")

    def s_lmn(l, m, n, sigma):
        """
        Shape of the PSF 3D base on a lorentzian distribution
        Parameters :
        * `sigma` : standard deviation 
        * `l,m,n` : position parameters
        """
        return (1/pi)*(gamma_var/(l**2+gamma_var**2))*np.exp(-(m**2+n**2)/(2*sigma**2))/(2*pi*sigma**2)

    def kernel(a, L, T): 
        """
        Calculation of the 3D Kernel for a gamma distribution background
        Parameters :
        * `a` : signal value
        * `L` : spatial length size
        * `T` : frequency length size
        """
        m = np.arange(-L,L+1).reshape(2*L+1,1)
        n = np.arange(-L,L+1).reshape(1,2*L+1)
        M = np.repeat(m, 2*L+1 ,1)
        N = np.repeat(n, 2*L+1 ,0)
        O = np.arange(-T, T+1)
        K = a * s_lmn(O[:,np.newaxis,np.newaxis], M[np.newaxis,:], N[np.newaxis,:], sigma)
        return K

    def tau(a, alpha, p_fp, L, T):
        """ 
        threshold calcutation function for the gamma distribution background
        Parameters:
        * `a`: signal value
        * `alpha`: alpha matrix
        * `P_FP`: probability of false positive
        * `L`: spatial length size
        * `T`: frequency length size
        """
        W1 = kernel(a, L, T)
        W2 = W1**2
        w1 = W1.sum()
        w2 = W2.sum()
        return np.sqrt(w2*trigamma(alpha))*inv_gauss(p_fp)+w1*digamma(alpha)

    def thresholding(mat_full_image):
        """
        thresholding 3D algorithm using an adaptive background 
        Parameter:
        * `mat_full_image` : 3D matrix given to get its thresholding 3D matrix
        

        # fitting backgrounds parameters
        
        estimation of the background parameters

        # parameters
        
        calculation of the background value b 
        and the signal value a
        
        # filtering process
        
        choice of the size parameters for the kernels and application of the functions
        * be aware, if you change the spatial dimension size, you have to change the background kernel to match the sizes
        * if there is some values equal to 0 we counter this by changing the values by 1

        `kernel_size_l` : frequency dimension

        `kernel_size_mn` : spatial dimensions


        
        # adaptive background
        
        application of the background kernel to get our adaptive background
        * be aware, if you change the spatial dimension size, you have to change the background kernel to match the sizes

        # Thresholding
        
        threshold process by calculating the tau matrix and comparing with the filtered matrix
        then returning the coordinates and the filtered matrix
        
        """

    def local_maxima_function(coordinates,filtered_mat):
        """
        local maxima detection algorithm
        parameters:
        * `coordinates` : coordinates of the potential maxima 
        * `filtered_mat` : filtered matrix 

        # init
        the values of the potential maxima are recovered and sorted
        then the coordinates are sorted
        is_local_max : array use to validate or no the maxima in the algorithm part
        

        # algorithm

        is_local_max is explored, 
        if the explored value is 0 then the algorithm is checking in the neighborhood (distance d under 3*sigma) if the value is a maxima then it assign a score :
        * 1 if it is local maxima
        * -1 if it is not
        * the neighbor also get a score of -1 if they are under the values check 

        the algorithm returns the local maxima coordinates
        """

    ### multiprocessing ###
    
    """
    multiprocessing execution based on a 1 producer / multiple consumers model.
    
    """
    
    @dataclass
    class threshold_result_data:
        """
        dataclass to return the resulted data from the threshold and local maxima algorithms
        
        `file_name`: str

        `fwhm`: float

        `probability_false_positive` : float

        `coordinates_array`: np.ndarray
        """
    
    def slicer(queue, nb_workers, nb_seg_part, file):
        """
        slicing of the video and putting the sliced parts in a queue to be processed by the workers
        parameters:
        * `queue` : get the sliced parts
        * `nb_workers` : get the number of workers to know how many stop signal (None) are needed
        * `nb_seg_part` : number of sliced parts wanted
        * `file` : name of the file wanted
        """
        time_start = time.perf_counter()
        print("[slicer] Starting execution...")
        tif = TiffFile(file)
        file_size = len(tif.pages)
        part = file_size // nb_seg_part
    
        for i in range(nb_seg_part):
            if i+1 == nb_seg_part:
                mat_segmented = imread(file,key = range(part*i,file_size))
            else:
                mat_segmented = imread(file,key = range(part*i,part*(i+1)))
            queue.put((mat_segmented,i,file_size))
        print("[slicer] Finished creating tasks. Sending shutdown signals...")
        for _ in range(nb_workers):
            queue.put(None)
        time_stop = time.perf_counter()
        execution_time = time_stop - time_start
        print(f"[slicer] Finished execution in {execution_time}s.")
    
    
    def worker(queue_in, queue_out, worker_id):
        """
        processing the parts received by the slicer and executing the threshold and the local_maxima algorithms
        parameters:
        * `queue_in` : queue from the slicer to get the sliced parts
        * `queue_out` : queue from the workers to transfer the coordinates to the builder
        * `worker_id` : use to know which worker is working
        """
        time_start = time.perf_counter()
        print(f"[worker-{worker_id}] Starting execution...")
        cnt_item_processed = 0
        while True:
            item = queue_in.get()
            
            if item is None:
                print(f"[worker-{worker_id}] Received shutdown signal. Exiting.")
                break
    
            mat_segmented, id, file_size = item
            coordinates,filtered_mat = thresholding(mat_segmented)
            local_maxima_coords = local_maxima_function(coordinates,filtered_mat)
            queue_out.put((local_maxima_coords,id,file_size))
            cnt_item_processed += 1
        
        queue_out.put(None)
        time_stop = time.perf_counter()
        execution_time = time_stop - time_start
        print(f"number of task count processed by the worker - {worker_id} : {cnt_item_processed}, in {execution_time}")
    
    
    def builder(queue_in,nb_seg_part, nworkers, queue_out, file):
        """
        concatenating all the coodinates found by the workers into one file and remove the multiple occurences
        parameters:
        * `queue_in` : queue with the coordinates from the workers
        * `nb_seg_part` : number of sliced parts wanted
        * `nworkers` : number of workers
        * `queue_out` : recover the data with the coordinates
        * `file` : name of the file, used in the dataclass with the data
        """
        time_start = time.perf_counter()
        print("[builder] Starting execution...")
        l = []
        cnt_item_processed = 0
        stop_count = nworkers
        while True:
            item = queue_in.get()
            
            if item is None:
                stop_count -= 1
    
                if stop_count <= 0:
                    print(f"[builder Received shutdown signal. Exiting.")
                    break
            else:
                coords, id, file_size = item
                part = file_size // nb_seg_part
                coords_mod = coords[0] + part*id, coords[1], coords[2]
                l.append(np.array(coords_mod))
                cnt_item_processed +=1
    
        f,x,y = np.concat(l, axis=1)
        coords = np.column_stack((x, y))
        _, index = np.unique(coords, axis=0, return_index=True)
        index = np.sort(index)
        x_uniques = x[index]
        y_uniques = y[index]
        f_uniques = f[index]
        result_array = np.array([f_uniques,x_uniques,y_uniques])
        print(result_array.shape)
        data = threshold_result_data(file_name=file, fwhm= gamma_var, probability_false_positive= p_fp, coordinates_array= result_array)
        queue_out.put(data)
        queue_out.put(None)
        print(f"nb item cnt processed by the builder:{cnt_item_processed}")
        time_stop = time.perf_counter()
        execution_time = time_stop - time_start
        print(f"[builder] Finished execution in {execution_time}")
