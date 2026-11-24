import numpy as np

## manually thresholding

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

file = 'C:/Users/Maelig/Documents/Research/molecule_detection/research/tiff_files/long_cool_scan.tif'
mat = imread(file)

m = np.mean(mat)

std = np.std(mat)

threshold = m + 7*std

T = mat > threshold
# print(np.nonzero(T[:,:,1:]))
# for i in range(0,10):
#     plt.matshow(T[i])
#     plt.show()

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

# print(np.shape(T[0]))
# print(np.shape(mat))
# print(local_maxima_function(np.nonzero(T[:,:,1:]),mat))

local_max_coordinates = local_maxima_function(np.nonzero(T[:,:,:498]),mat[:,:,:498])
permutation2 = np.argsort(local_max_coordinates[0])[::]
local_max_coordinates_sorted = local_max_coordinates[0][permutation2],local_max_coordinates[1][permutation2],local_max_coordinates[2][permutation2]

mat_display = np.load('filtered_test_mat.npy')
start=0
end=40
c=0
for i in range(start,end):
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.matshow(mat_display[i])
    while i == local_max_coordinates_sorted[0][c]:
        plt.scatter(local_max_coordinates_sorted[2][c],local_max_coordinates_sorted[1][c], s=1000, facecolors='none', edgecolors='r', lw=1)
        c=c+1
    plt.show()