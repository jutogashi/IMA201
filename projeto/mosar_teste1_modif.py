# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 16:09:54 2020

@author: Filipe
"""


import matplotlib.pyplot as plt
#from skimage import data
from skimage import io as skio
#from skimage.filters import threshold_otsu
import numpy as np
#from skimage.morphology import binary_dilation
#import matplotlib.pyplot as plt
from scipy.io.matlab.mio import loadmat
#from colorsys import hsv_to_rgb
from scipy import special
path = 'file:///C:/Users/jutogashi/Documents/Telecom_2A/IMA/IMA201/projeto/DonneesRivieresSAR2SAR/'


def histogram(im):
    
    nl,nc=im.shape
    
    hist=np.zeros(256)
    
    for i in range(nl):
        for j in range(nc):
            hist[im[i][j]]=hist[im[i][j]]+1
            
    for i in range(256):
        hist[i]=hist[i]/(nc*nl)
        
    return(hist)

def min_err_t(h):        
    
    mint=0
    mink=np.inf
    #J=[]
    
    for t in range(256):
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
            v0=v0+(h[i]*(i - m0)**2)
        if p0 > 0:
            v0=v0/p0
        
        for i in range(t,256):
            p1=p1+h[i]
            m1=m1+i*h[i]
        if p1 > 0:   
            m1=m1/p1
        
        for i in range(t,256):
            v1=v1+(h[i]*(i - m1)**2)
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
    
 
    
    
    
def min_err_t_wibull(im):     
    
    h= histogram(im)
    nl,nc=im.shape
    x0=0
    y0=0
    z0=0
    x1=0
    y1=0
    z1=0
    
    mint=0
    mink=np.inf
    
    for t in range(256):
        P0=0
        P1=0
        k=np.inf
        
        for i in range(t):
            if i>0:
                P0=P0+h[i]
                x0 = x0 + h[i]*np.log(i)
                y0 = y0 + h[i]
        
        for i in range(t,256):
            if i>0:
                P1=P1+h[i]
                x1 = x1 + h[i]*np.log(i)
                y1 = y1 + h[i]
        
        if(y0>0):
            k10=x0/y0
        if(y1>0):
            k11=x1/y1
        
        if(y0 >0 and y1 >0): 
            for i in range(t):
                if i>0:
                    z0 = z0 + h[i]*((np.log(i)-k10)**2)
                
            for i in range(t,256):
                if i>0:
                    z1 = z1 + h[i]*((np.log(i)-k11)**2)
                    
            k20=z0/y0
            k21=z1/y1
            
        k=mink    
        if(y0 >0 and y1 >0):
            lamb0 = np.exp(k10)
            n0 = np.sqrt(2*special.polygamma(1,1)/k20)
            
            lamb1 = np.exp(k11)
            n1 = np.sqrt(2*special.polygamma(1,1)/k21)
            
            k0 = (P0 * np.log(P0))
            for i in range(t):
                p0= n0*lamb0**n0*((i**(n0-1))/((lamb0**n0)+(i**n0))**2)
                k0 = k0+ h[i]*np.log(p0)
                
            k1 = (P1 * np.log(P1))  
            for i in range(t,256):
                p1= n1*lamb1**n1*((i**(n1-1))/((lamb1**n1)+(i**n1))**2)
                k1 = k1 + h[i]*np.log(p1)
                
            k=-(k0+k1)
            print(k)
        
        if k < mink:
            mink=k
            mint=t
                       
    thresh=mint
        
    return(thresh)    
    
    
    
def loadImageFromMatFile(file, date = 0):
    mat = loadmat(file)
    vh = np.array(mat['images_choisiesVH'], np.double)[:,:,date]
    vv = np.array(mat['images_choisiesVV'], np.double)[:,:,date]
    truth = np.array(mat['array_segmentation'], np.double)[:,:,date]
    return vh, vv, truth


def image_t(im):
    h = histogram(im)
    thresh=min_err_t(h)
    binary = im > thresh
    
    return binary

#noise
noise1 = skio.imread(path + 'noisy_MadridIowa_dual_1_corrige_1_VH_AMPLITUDE_date_0.png')
noise2 = skio.imread(path + 'noisy_MadridIowa_dual_1_corrige_1_VH_AMPLITUDE_date_2.png')

#rapport
image = np.uint8(noise1/noise2)

#difference
#image = np.uint8(noise1 - noise2)

#log
c = 255/(np.log(1 + np.max(noise1/(noise2+0.0001)))) 
log_transformed = c * np.log(1 + noise1/noise2) 
image = np.uint8(log_transformed)

h = histogram(image)
thresh=min_err_t_wibull(image)
binary = image > thresh


#image histogram with threhshold
bins=np.max(image)-np.min(image)+1
plt.figure()
plt.hist(image.ravel(), bins=bins)
plt.axvline(thresh, color='r')
plt.title('Histogram with the threhsold')
plt.show()

plt.figure()
plt.imshow(binary, cmap='gray')
plt.title('Minimum error thresholding')
plt.show()

#############################################################################################

#denoise
denoise1 = skio.imread(path + 'denoised_MadridIowa_dual_1_corrige_1_VH_AMPLITUDE_date_0.png')
denoise2 = skio.imread(path + 'denoised_MadridIowa_dual_1_corrige_1_VH_AMPLITUDE_date_2.png')

#rapport
image = np.uint8(denoise1/denoise2)

#difference
#image = np.uint8(denoise1 - denoise2)

#log
c = 255/(np.log(1 + np.max(denoise1/denoise2))) 
log_transformed = c * np.log(1 + denoise1/denoise2) 
image = np.uint8(log_transformed)

h = histogram(image)
thresh=min_err_t(h)
binary = image > thresh


#image histogram with threhshold
bins=np.max(image)-np.min(image)+1
plt.figure()
plt.hist(image.ravel(), bins=bins)
plt.axvline(thresh, color='r')
plt.title('Histogram with the threhsold')
plt.show()


plt.figure()
plt.imshow(binary, cmap='gray')
plt.title('Minimum error thresholding')
plt.show()





