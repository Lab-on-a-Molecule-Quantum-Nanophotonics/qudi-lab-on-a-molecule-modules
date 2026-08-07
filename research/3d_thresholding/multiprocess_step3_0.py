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

pi=np.pi
sigma=(1/2.3548) * 2.7 # Calculated in the dedicated notebook
gamma_var = 2.7 # fwhm Calculated in the dedicated notebook
p_fp = 1e-6
    ### Functions ###



def ln_transform(r_mn,theta):
    return np.log(r_mn, dtype="float16")-np.log(theta, dtype="float16") # modification of the dtype for memory/speed issues


def s_mn(l, m, n, sigma):
    return (1/pi)*(gamma_var/(l**2+gamma_var**2))*np.exp(-(m**2+n**2)/(2*sigma**2))/(2*pi*sigma**2)


def kernel(a, L, T): 
    m = np.arange(-L,L+1).reshape(2*L+1,1)
    n = np.arange(-L,L+1).reshape(1,2*L+1)
    M = np.repeat(m, 2*L+1 ,1)
    N = np.repeat(n, 2*L+1 ,0)
    O = np.arange(-T, T+1)
    K = a * s_mn(O[:,np.newaxis,np.newaxis], M[np.newaxis,:], N[np.newaxis,:], sigma)
    return K


def digamma(z):
    return polygamma(0,z)


def trigamma(z):
    return polygamma(1,z)


def inv_gauss(x):
    return norm.isf(x, loc=0, scale=1)


def tau(a, alpha, p_fp, L, T):
    """ 
    Parameters:
    * a: signal value
    * alpha: alpha matrix
    * P_FP: probability of false positive
    * L: kernel size
    * T: kernel freq size
    """
    W1 = kernel(a, L, T)
    W2 = W1**2
    w1 = W1.sum()
    w2 = W2.sum()
    return np.sqrt(w2*trigamma(alpha))*inv_gauss(p_fp)+w1*digamma(alpha)


def thresholding(mat_full_image):
    
    # fitting backgrounds parameters 
    raveled_mat=np.ravel(mat_full_image[0])
    raveled_mat_bg = raveled_mat[raveled_mat<np.quantile(mat_full_image, 1)]

    alpha, loc, theta = gamma.fit(raveled_mat_bg, floc=0.0)

    # print(f"Fit result alpha={alpha}, loc={loc}, theta={theta}")

    # parameters
    b = alpha*theta #background value: alpha*theta
    a = np.max(mat_full_image[mat_full_image < np.quantile(mat_full_image, 0.99999)]) - b


    # filtering process
    N = 5
    kernel_size = 3 ## if it is change, you have to also change K (bg kernel) to match the sizes


    ker = kernel(a, kernel_size, N)
    mat_full_image[np.where(mat_full_image == 0)] = 1 # prevent to have log of 0 

    log_transformed_mat = ln_transform(mat_full_image,theta)
    # print("The size of the log transformed image is", log_transformed_mat.shape)

    filtered_mat = convolve(log_transformed_mat, ker, mode='valid')
    # print("The size of the filtered image is", filtered_mat.shape)

    # adaptive background
    K=(1/(40*(2*N+1)))*np.array([[[1,1,1,1,1,1,1],
                              [1,1,1,1,1,1,1],
                              [1,1,0,0,0,1,1],
                              [1,1,0,0,0,1,1],
                              [1,1,0,0,0,1,1],
                              [1,1,1,1,1,1,1],
                              [1,1,1,1,1,1,1]]]*(2*N+1)) ## warning : the size of K must matched the kernel_size to make the bg and the filtered image sizes matched

    bg_adaptive = convolve(mat_full_image, K, mode='valid')
    # print("The size of the bg element image is", bg_adaptive.shape)

    # Thresholding
    Alpha = bg_adaptive/theta

    tau_mat = tau(a,Alpha, p_fp, kernel_size, N)
    # print("The size of the tau image is", tau_mat.shape)

    T = filtered_mat >= tau_mat
    coordinates = np.nonzero(T[:,:,1:])
    return coordinates, filtered_mat


def local_maxima_function(coordinates,filtered_mat):

    # init
    values = filtered_mat[coordinates[0],coordinates[1],coordinates[2]]
    permutation = np.argsort(values)[::-1]
    coordinates_sorted = coordinates[0][permutation],coordinates[1][permutation],coordinates[2][permutation]
    # values_sorted = values[permutation]
    # print(np.max(values),values_sorted)
    is_local_max=np.full(len(coordinates[0]),0)

    # algo
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



@dataclass
class threshold_result_data:
    file_name: str
    fwhm: float
    probability_false_positive : float
    coordinates_array: np.ndarray


def slicer(queue, nb_workers,nb_seg_part, file):
    time_start = time.perf_counter()
    print("[slicer] Starting execution...")
    tif = TiffFile(file)
    file_size = len(tif.pages)
    part = file_size // nb_seg_part
    remain = file_size % nb_seg_part

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
    time_start = time.perf_counter()
    print(f"[worker-{worker_id}] Starting execution...")
    cnt_item_processed = 0
    while True:
        item = queue_in.get()
        
        if item is None:
            print(f"[worker-{worker_id}] Received shutdown signal. Exiting.")
            break

        mat_segmented, id, file_size = item
        # print(f"[worker-{worker_id}] Processing...")
        coordinates,filtered_mat = thresholding(mat_segmented)
        local_maxima_coords = local_maxima_function(coordinates,filtered_mat)
        queue_out.put((local_maxima_coords,id,file_size))
        cnt_item_processed += 1
        # print(local_maxima_coords)
        # print(f"[worker-{worker_id}] Completed.")
    
    queue_out.put(None)
    time_stop = time.perf_counter()
    execution_time = time_stop - time_start
    print(f"number of task count processed by the worker - {worker_id} : {cnt_item_processed}, in {execution_time}")


def builder(queue_in,nb_seg_part, nworkers, queue_out, file):
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

    result_array = np.concat(l, axis=1)
    data = threshold_result_data(file_name=file, fwhm= gamma_var, probability_false_positive= p_fp, coordinates_array= result_array)
    queue_out.put(data)
    queue_out.put(None)
    print(f"nb item cnt processed by the builder:{cnt_item_processed}")
    # print(result_array, type(result_array))
    time_stop = time.perf_counter()
    execution_time = time_stop - time_start
    print(f"[builder] Finished execution in {execution_time}")



if __name__ == "__main__":
    nb_seg_part = 64
    nb_workers = 8
    file = 'tiff_files/long_cool_scan.tif'
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

    print("Main process successfully finalized.")


