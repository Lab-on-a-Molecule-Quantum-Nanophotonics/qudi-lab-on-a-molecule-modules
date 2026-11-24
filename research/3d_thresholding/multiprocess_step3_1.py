    # libraries

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



pi=np.pi
gamma_var = 2.7 # fwhm Calculated in the dedicated notebook 
sigma=(1/2.3548) * gamma_var # standard deviation for a gaussian beam, Calculated in the dedicated notebook
p_fp = 1e-6

    ### Functions ###



def ln_transform(r_mn,theta):
    """
    log transform function use for the gamma distribution background
    Parameters :
    * r_mn : array given to get its log transform
    * theta : scale parameter from the gamma distribution
    """
    return np.log(r_mn, dtype="float16")-np.log(theta, dtype="float16")


def s_lmn(l, m, n, sigma):
    """
    Shape of the PSF 3D base on a lorentzian distribution
    Parameters :
    * sigma : standard deviation 
    * l,m,n : position parameters
    """
    return (1/pi)*(gamma_var/(l**2+gamma_var**2))*np.exp(-(m**2+n**2)/(2*sigma**2))/(2*pi*sigma**2)


def kernel(a, L, T): 
    """
    Calculation of the 3D Kernel for a gamma distribution background
    Parameters :
    * a : signal value
    * L : spatial length size
    * T : frequency length size
    """
    m = np.arange(-L,L+1).reshape(2*L+1,1)
    n = np.arange(-L,L+1).reshape(1,2*L+1)
    M = np.repeat(m, 2*L+1 ,1)
    N = np.repeat(n, 2*L+1 ,0)
    O = np.arange(-T, T+1)
    K = a * s_lmn(O[:,np.newaxis,np.newaxis], M[np.newaxis,:], N[np.newaxis,:], sigma)
    return K


def digamma(z):
    return polygamma(0,z)


def trigamma(z):
    return polygamma(1,z)


def inv_gauss(x):
    return norm.isf(x, loc=0, scale=1)


def tau(a, alpha, p_fp, L, T):
    """ 
    threshold calcutation function for the gamma distribution background
    Parameters:
    * a: signal value
    * alpha: alpha matrix
    * P_FP: probability of false positive
    * L: spatial length size
    * T: frequency length size
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
    * mat_full_image : 3D matrix given to get its thresholding 3D matrix
    """

    # fitting backgrounds parameters
    """
    estimation of the background parameters
    """
    raveled_mat=np.ravel(mat_full_image[0])
    raveled_mat_bg = raveled_mat[raveled_mat<np.quantile(mat_full_image, 1)]

    alpha, loc, theta = gamma.fit(raveled_mat_bg, floc=0.0)


    # parameters
    """
    calculation of the background value b 
    and the signal value a
    """
    b = alpha*theta
    a = np.max(mat_full_image[mat_full_image < np.quantile(mat_full_image, 0.99999)]) - b


    # filtering process
    """
    choice of the size parameters for the kernels and application of the functions
    * be aware, if you change the spatial dimension size, you have to change the background kernel to match the sizes
    * if there is some values equal to 0 we counter this by changing the values by 1
    kernel_size_l : frequency dimension
    kernel_size_mn : spatial dimensions
    """
    kernel_size_l = 5
    kernel_size_mn = 3


    ker = kernel(a, kernel_size_mn, kernel_size_l)
    mat_full_image[np.where(mat_full_image == 0)] = 1 # prevent to have log of 0 

    log_transformed_mat = ln_transform(mat_full_image,theta)

    filtered_mat = convolve(log_transformed_mat, ker, mode='valid')

    # adaptive background
    """
    application of the background kernel to get our adaptive background
    * be aware, if you change the spatial dimension size, you have to change the background kernel to match the sizes
    """
    background_kernel=(1/(40*(2*kernel_size_l+1)))*np.array([[[1,1,1,1,1,1,1],
                                                            [1,1,1,1,1,1,1],
                                                            [1,1,0,0,0,1,1],
                                                            [1,1,0,0,0,1,1],
                                                            [1,1,0,0,0,1,1],
                                                            [1,1,1,1,1,1,1],
                                                            [1,1,1,1,1,1,1]]]*(2*kernel_size_l+1))

    bg_adaptive = convolve(mat_full_image, background_kernel, mode='valid')

    # Thresholding
    """
    threshold process by calculating the tau matrix and comparing with the filtered matrix
    then returning the coordinates and the filtered matrix
    """
    alpha = bg_adaptive/theta

    tau_mat = tau(a,alpha, p_fp, kernel_size_mn, kernel_size_l)

    T = filtered_mat >= tau_mat
    coordinates = np.nonzero(T[:,:,1:])

    return coordinates, filtered_mat


def local_maxima_function(coordinates,filtered_mat):
    """
    local maxima detection algorithm
    parameters:
    * coordinates : coordinates of the potential maxima 
    * filtered_mat : filtered matrix 
    """
    # init
    """
    the values of the potential maxima are recovered and sorted
    then the coordinates are sorted
    is_local_max : array use to validate or no the maxima in the algorithm part
    """
    values = filtered_mat[coordinates[0],coordinates[1],coordinates[2]]
    permutation = np.argsort(values)[::-1]
    coordinates_sorted = coordinates[0][permutation],coordinates[1][permutation],coordinates[2][permutation]
    is_local_max=np.full(len(coordinates[0]),0)

    # algorithm
    """
    is_local_max is explored, 
    if the explored value is 0 then the algorithm is checking in the neighborhood (distance d under 3*sigma) if the value is a maxima then it assign a score :
    * 1 if it is local maxima
    * -1 if it is not
    * the neighbor also get a score of -1 if they are under the values check 

    the algorithm returns the local maxima coordinates
    """
    l = len(coordinates[0])
    for i in range(l):
        if is_local_max[i] == 0 :
            # coordinates and distance
            l,m,n = coordinates_sorted[0][i],coordinates_sorted[1][i],coordinates_sorted[2][i]
            d = np.sqrt((coordinates_sorted[0]-l)**2 + (coordinates_sorted[1]-m)**2 + (coordinates_sorted[2]-n)**2)

            # neighborhood mask
            roi = (0 < d) & (d < 3*sigma)
            neighbor_coord = coordinates_sorted[0][roi],coordinates_sorted[1][roi],coordinates_sorted[2][roi]

            test = filtered_mat[l][m][n] > filtered_mat[neighbor_coord]
            local_max_test = np.all(test)

            is_local_max[roi] = np.where(test,-1,is_local_max[roi])
            if local_max_test == True :
                is_local_max[i] = 1
            else :
                is_local_max[i] = -1

    else :
        is_local_max[i] = -1

    local_max_mask = is_local_max == 1

    local_max_coordinates = coordinates_sorted[0][local_max_mask],coordinates_sorted[1][local_max_mask],coordinates_sorted[2][local_max_mask]
    
    return local_max_coordinates



    ### multiprocessing ###

"""
multiprocessing execution based on a 1 producer / multiple consumers model.

"""

@dataclass
class threshold_result_data:
    """
    dataclass to return the resulted data from the threshold and local maxima algorithms
    """
    file_name: str
    fwhm: float
    probability_false_positive : float
    coordinates_array: np.ndarray


def slicer(queue, nb_workers, nb_seg_part, file):
    """
    slicing of the video and putting the sliced parts in a queue to be processed by the workers
    parameters:
    * queue : get the sliced parts
    * nb_workers : get the number of workers to know how many stop signal (None) are needed
    * nb_seg_part : number of sliced parts wanted
    * file : name of the file wanted
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
    * queue_in : queue from the slicer to get the sliced parts
    * queue_out : queue from the workers to transfer the coordinates to the builder
    * worker_id : use to know which worker is working
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
    * queue_in : queue with the coordinates from the workers
    * nb_seg_part : number of sliced parts wanted
    * nworkers : number of workers
    * queue_out : recover the data with the coordinates
    * file : name of the file, used in the dataclass with the data
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



if __name__ == "__main__":
    """
    choice of the number of parts and workers wanted depending on the number of cores
    * it is better to choose a highest number of parts compared to the number of workers
    """
    nb_seg_part = 64
    nb_workers = 8
    file = "H:/Other computers/Spectro2021/Lab66_spectrometer_2021/Data/2025/05/2025-05-23/long_cool_scan_wf_5uW_res_3p8K_2by2_10xGain.tif"
    queue_in = Queue()
    queue_out = Queue()
    queue_result = Queue()

    slicing_process = Process(target=slicer, args=(queue_in, nb_workers, nb_seg_part, file))

    worker_processes = []
    for i in range(nb_workers):
        working_process = Process(target=worker, args=(queue_in, queue_out,i))
        worker_processes.append(working_process)
    
    builder_process = Process(target=builder, args=(queue_out, nb_seg_part, nb_workers, queue_result, file))

    slicing_process.start()
    for working_process in worker_processes:
        working_process.start()
    builder_process.start()
    

    slicing_process.join()

    for working_process in worker_processes:
        working_process.join()

    builder_process.join()
    threshold_coordinates = queue_result.get()
    if threshold_coordinates is not None:
        print(threshold_coordinates)
    else:
        print('error, no item in the queue')


    ### diplaying of the result

    
    # mat = np.load('C:/Users/Maelig/Documents/Research/molecule_detection/research/filtered_test_mat.npy')
    local_max_coordinates = threshold_coordinates.coordinates_array
    permutation2 = np.argsort(local_max_coordinates[0])[::]
    local_max_coordinates_sorted = local_max_coordinates[0][permutation2],local_max_coordinates[1][permutation2],local_max_coordinates[2][permutation2]
    print(local_max_coordinates_sorted)

    start=0
    end=40
    c=0
    for i in range(start,end):
        fig, ax = plt.subplots(figsize=(12, 12))
        # ax.matshow(mat[i])
        while i == local_max_coordinates_sorted[0][c]:
            plt.scatter(local_max_coordinates_sorted[2][c],local_max_coordinates_sorted[1][c], s=1000, facecolors='none', edgecolors='r', lw=1)
            c=c+1
        plt.show()
    print("Main process successfully finalized.")

    # c=0
    # data = np.load('filtered_test_mat.npy')
    # fig, ax = plt.subplots()
    # ims = []
    # ims2 = []
    # for i in range(data.shape[0]):
    #     while i == local_max_coordinates_sorted[0][c]:
    #         im = ax.imshow(data[i], cmap="viridis", animated=True)
    #         ims2.append([im])
    #         im = ax.scatter(local_max_coordinates_sorted[2][c],local_max_coordinates_sorted[1][c], s=1000, facecolors='none', edgecolors='r', lw=1)
    #         c=c+1
    #         ims.append([im])

    # ani = animation.ArtistAnimation(
    #     fig, ims, interval=100, blit=True
    # )

    # plt.show()
