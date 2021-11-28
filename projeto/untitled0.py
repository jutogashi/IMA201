# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 12:19:23 2020

@author: jutogashi
"""

# -- coding: utf-8 --
"""
Created on Sun Oct 25 16:09:54 2020

@author: Filipe
"""

import matplotlib.pyplot as plt
#from skimage import data
from skimage import io as skio
#from skimage.filters import threshold_otsu
import numpy as np
#from colorsys import hsv_to_rgb
path = 'file:///C:/Users/jutogashi/Documents/Telecom_2A/IMA/IMA201/projeto/DonneesRivieresSAR2SAR/'



def otsu_thresh(h):
    
    m=0
    for i in range(len(h)):
        m=m+i*h[i]
    
    maxt=0
    maxk=0
    
    
    for t in range(len(h)):
        w0=0
        w1=0
        m0=0
        m1=0
        for i in range(t):
            w0=w0+h[i]
            m0=m0+i*h[i]
        if w0 > 0:
            m0=m0/w0
        
        for i in range(t,len(h)):
            w1=w1+h[i]
            m1=m1+i*h[i]
        if w1 > 0:   
            m1=m1/w1
        
        k=w0*w1*(m0-m1)*(m0-m1)    
        
        if k > maxk:
            maxk=k
            maxt=t
            
            
    thresh=maxt
        
    return(thresh)

def min_err_t(h):        
    
    mint=0
    mink=np.inf
    #J=[]
    
    
    for t in range(len(h)):
        p0=0
        p1=0
        m0=0
        m1=0
        v0=0
        v1=0
        k=np.inf
        
        for i in range(t):
            p0=p0+h[i]
            m0=m0+i*h[i]
        if p0 > 0:
            m0=m0/p0
            
        for i in range(t):
            v0=v0+(h[i](i - m0)*2)
        if p0 > 0:
            v0=v0/p0
        
        for i in range(t,len(h)):
            p1=p1+h[i]
            m1=m1+i*h[i]
        if p1 > 0:   
            m1=m1/p1
        
        for i in range(t,len(h)):
            v1=v1+(h[i](i - m1)*2)
        if p1 > 0:
            v1=v1/p1
        if v0 > 0 and v1 >0:
            k = 1 + 2 * (p0 * np.log(v0) + p1 * np.log(v1)) - 2 * (p0 * np.log(p0) + p1 * np.log(p1))
        
        #J.append(k)
        
        if k < mink:
            mink=k
            mint=t
            
            
    thresh=mint
        
    return(thresh)



#noise
    

noise1 = np.load('noisy_MadridIowa_dual_1_corrige_1_VH_AMPLITUDE_date_0.npy')
noise2 = np.load('noisy_MadridIowa_dual_1_corrige_1_VH_AMPLITUDE_date_2.npy')


#noise1 = skio.imread('noisy_MadridIowa_dual_1_corrige_1_VH_AMPLITUDE_date_0.png')
#noise2 = skio.imread( 'noisy_MadridIowa_dual_1_corrige_1_VH_AMPLITUDE_date_2.png')


#rapport
image = noise1/noise2


#difference en log
image = np.log(noise1)- np.log(noise2)
image = np.log(noise1/noise2)

mask = np.zeros(image.shape)
mask[410:,750:] = 1

#th = 200
#n_bins=int(np.floor(np.max(image[image<th])-np.min(image)+1))
#h = np.histogram(image, bins=n_bins)[0]

n_bins=100
h = np.histogram(image, range = (np.min(image),np.max(image)), bins=n_bins)

t=min_err_t(h[0])
thresh_min = h[1][t]
min_binary = image > thresh_min

t=otsu_thresh(h[0])
thresh_otsu = h[1][t]
otsu_binary = image > thresh_otsu


#image histogram with threhshold

plt.figure()
plt.hist(image.ravel(), bins=n_bins)
plt.axvline(thresh_min, color='b', label='min error threhsold')
plt.axvline(thresh_otsu, color='r', label='threhsold otsu')
plt.title('Histogram with the threhsold')
plt.legend()
plt.show()

plt.figure()
plt.imshow(min_binary, cmap='gray')
plt.title('Minimum error thresholding without mask')
plt.show()

plt.figure()
plt.imshow(otsu_binary, cmap='gray')
plt.title('Otsu threshold without mask')
plt.show()

# mask

plt.figure()
plt.imshow(mask*otsu_binary, cmap='gray')
plt.title('Otsu threshold with mask')
plt.show()

plt.figure()
plt.imshow(mask*min_binary, cmap='gray')
plt.title('Minimum error thresholding with mask')
plt.show()


#############################################################################################

#denoise

denoise1 = np.load('denoised_MadridIowa_dual_1_corrige_1_VH_AMPLITUDE_date_0.npy')
denoise2 = np.load('denoised_MadridIowa_dual_1_corrige_1_VH_AMPLITUDE_date_2.npy')

#rapport
image = denoise1/denoise2

#difference
#image = denoise1 - denoise2

mask = np.zeros(image.shape)
mask[410:,750:] = 1


n_bins=int(np.floor(np.max(image)-np.min(image)+1))
h = np.histogram(image, bins=n_bins)

n_bins=1000
h = np.histogram(image, range = (np.min(image),np.max(image)), bins=n_bins)

t=min_err_t(h[0])
thresh_min = h[1][t]
min_binary = image > thresh_min

t=otsu_thresh(h[0])
thresh_otsu = h[1][t]
otsu_binary = image > thresh_otsu


#image histogram with threhshold

plt.figure()
plt.hist(image.ravel(), bins=n_bins)
plt.axvline(thresh_min, color='b', label='min error threhsold')
plt.axvline(thresh_otsu, color='r', label='threhsold otsu')
plt.title('Histogram with the threhsold')
plt.legend()
plt.show()

plt.figure()
plt.imshow(min_binary, cmap='gray')
plt.title('Minimum error thresholding without mask')
plt.show()

plt.figure()
plt.imshow(otsu_binary, cmap='gray')
plt.title('Otsu threshold without mask')
plt.show()

# mask

plt.figure()
plt.imshow(mask*otsu_binary, cmap='gray')
plt.title('Otsu threshold with mask')
plt.show()

plt.figure()
plt.imshow(mask*min_binary, cmap='gray')
plt.title('Minimum error thresholding with mask')
plt.show()