    # libraries

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gamma as G
from scipy.special import polygamma
from scipy.stats import gamma,norm
from scipy.signal import convolve
from tifffile import imread

    # segmented video opening

mat_full_image = imread('C:/Users/Maelig/Documents/Research/molecule_detection/research/tiff_files/long_cool_scan.tif',key = range(0,200))
scan_shape=mat_full_image.shape

    # fitting backgrounds parameters 

raveled_mat=np.ravel(mat_full_image[0])
raveled_mat_bg = raveled_mat[raveled_mat<np.quantile(mat_full_image, 1)]

alpha, loc, theta = gamma.fit(raveled_mat_bg, floc=0.0)

print(f"Fit result alpha={alpha}, loc={loc}, theta={theta}")

    # fitting diplaying

# x = np.linspace(start=300, stop=1000, num=1000)
# pdf_fitted = gamma.pdf(x, a=alpha, loc=loc, scale=theta)

# # Plot histogram and fitted curve
# plt.hist(raveled_mat_bg, bins='auto', density=True, alpha=0.5, label='Data')
# plt.plot(x, pdf_fitted, 'r-', label='Fitted Gamma PDF')
# plt.xlim([300,1000])
# plt.title('Fitting Gamma Distribution to Data')
# plt.xlabel('x')
# plt.ylabel('Density')
# plt.legend()
# plt.grid(True)
# plt.show()

    ### Filtering

    # parameters

pi = np.pi
b = alpha*theta #background value: alpha*theta
a = np.max(mat_full_image[mat_full_image < np.quantile(mat_full_image, 0.99999)]) - b
sigma=(1/2.3548) * 2.7 # Calculated in the dedicated notebook
gamma = 2.7 # fwhm Calculated in the dedicated notebook

    # Functions

def ln_transform(r_mn,theta):
    return np.log(r_mn, dtype="float16")-np.log(theta, dtype="float16") # modification of the dtype for memory/speed issues

def s_mn(l, m, n, sigma):
    return (1/pi)*(gamma/(l**2+gamma**2))*np.exp(-(m**2+n**2)/(2*sigma**2))/(2*pi*sigma**2)

def Kernel(a, L, T): 
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

def tau(a, alpha, P_FP, L, T):
    """ 
    Parameters:
    * a: signal value
    * alpha: alpha matrix
    * P_FP: probability of false positive
    * L: kernel size
    * T: kernel freq size
    """
    W1 = Kernel(a, L, T)
    W2 = W1**2
    w1 = W1.sum()
    w2 = W2.sum()
    return np.sqrt(w2*trigamma(alpha))*inv_gauss(P_FP)+w1*digamma(alpha)

    # filtering process

N = 5
kernel_size = 3 ## if it is change, you have to also change K (bg kernel) to match the sizes


ker = Kernel(a, kernel_size, N)
mat_full_image[np.where(mat_full_image == 0)] = 1 # prevent to have log of 0 

log_transformed_mat = ln_transform(mat_full_image,theta)
print("The size of the log transformed image is", log_transformed_mat.shape)

filtered_mat = convolve(log_transformed_mat, ker, mode='valid')
print("The size of the filtered image is", filtered_mat.shape)
np.save('filtered_test_mat.npy',filtered_mat)
    ### Thresholding

    # adaptive background

K=(1/(40*(2*N+1)))*np.array([[[1,1,1,1,1,1,1],
                              [1,1,1,1,1,1,1],
                              [1,1,0,0,0,1,1],
                              [1,1,0,0,0,1,1],
                              [1,1,0,0,0,1,1],
                              [1,1,1,1,1,1,1],
                              [1,1,1,1,1,1,1]]]*(2*N+1)) ## warning : the size of K must matched the kernel_size to make the bg and the filtered image sizes matched

bg_adaptive = convolve(mat_full_image, K, mode='valid')
print("The size of the bg element image is", bg_adaptive.shape)

    # Thresholding

P_FP = 1e-6
Alpha = bg_adaptive/theta

tau_mat = tau(a,Alpha, P_FP, kernel_size, N)
print("The size of the tau image is", tau_mat.shape)

T = filtered_mat >= tau_mat

    ### Local max coordinates

    # init

coordinates = np.nonzero(T[:,:,1:])
values = filtered_mat[coordinates[0],coordinates[1],coordinates[2]]
permutation = np.argsort(values)[::-1]
coordinates_sorted = coordinates[0][permutation],coordinates[1][permutation],coordinates[2][permutation]
values_sorted = values[permutation]
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

print("local max coordinates = ", local_max_coordinates)
