# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 16:09:54 2020

@author: Filipe
"""
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
#from skimage import data
from skimage import io as skio
#from skimage.filters import threshold_otsu
import numpy as np
#from colorsys import hsv_to_rgb
from scipy import special
import scipy.stats as stats
from scipy.special import polygamma
from scipy.optimize import fsolve
import math
from scipy.io.matlab.mio import loadmat
import skimage.morphology as morpho  
path = 'file:///C:/Users/jutogashi/Documents/Telecom_2A/IMA/IMA201/projeto/DonneesRivieresSAR2SAR/'

def loadImageFromMatFile(file, date = 0):
    mat = loadmat(file)
    VH = np.array(mat['images_choisiesVH'], np.double)[:,:,date]
    VV = np.array(mat['images_choisiesVV'], np.double)[:,:,date]
    truth = np.array(mat['array_segmentation'], np.double)[:,:,date]
    return VH, VV, truth

def strel(forme,taille,angle=45):
    """renvoie un element structurant de forme  
     'diamond'  boule de la norme 1 fermee de rayon taille
     'disk'     boule de la norme 2 fermee de rayon taille
     'square'   carre de cote taille (il vaut mieux utiliser taille=impair)
     'line'     segment de langueur taille et d'orientation angle (entre 0 et 180 en degres)
      (Cette fonction n'est pas standard dans python)
    """

    if forme == 'diamond':
        return morpho.selem.diamond(taille)
    if forme == 'disk':
        return morpho.selem.disk(taille)
    if forme == 'square':
        return morpho.selem.square(taille)
    if forme == 'line':
        angle=int(-np.round(angle))
        angle=angle%180
        angle=np.float32(angle)/180.0*np.pi
        x=int(np.round(np.cos(angle)*taille))
        y=int(np.round(np.sin(angle)*taille))
        if x**2+y**2 == 0:
            if abs(np.cos(angle))>abs(np.sin(angle)):
                x=int(np.sign(np.cos(angle)))
                y=0
            else:
                y=int(np.sign(np.sin(angle)))
                x=0
        rr,cc=morpho.selem.draw.line(0,0,y,x)
        rr=rr-rr.min()
        cc=cc-cc.min()
        img=np.zeros((rr.max()+1,cc.max()+1) )
        img[rr,cc]=1
        return img
    raise RuntimeError('Erreur dans fonction strel: forme incomprise')
    


def min_err_t(h):        
    
    mint=0
    mink=np.inf
    J=[]
 
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
            v0=v0+(h[i]*(i - m0)**2)
        if p0 > 0:
            v0=v0/p0
        
        for i in range(t,len(h)):
            p1=p1+h[i]
            m1=m1+i*h[i]
        if p1 > 0:   
            m1=m1/p1
        
        for i in range(t,len(h)):
            v1=v1+(h[i]*(i - m1)**2)
        if p1 > 0:
            v1=v1/p1
        if v0 > 0 and v1 >0:
            k = 1 + 2 * (p0 * np.log(v0) + p1 * np.log(v1)) - 2 * (p0 * np.log(p0) + p1 * np.log(p1))
        
        J.append(k)
        
        if k < mink:
            mink=k
            mint=t
            
            
    thresh=mint
        
    return(thresh)
    
    

def min_err_t_lognorm(h):     
    
    
    mint=0
    mink=np.inf
    
    for t in range(1,len(h)):
        P0=0
        P1=0
        x0=0
        y0=0
        z0=0
        x1=0
        y1=0
        z1=0
        k=np.inf
        
        for u in range(1,t):
            P0=P0+h[u]
            x0 = x0 + h[u]*np.log(u)
            y0 = y0 + h[u]
        
        for u in range(t,len(h)):
            P1=P1+h[u]
            x1 = x1 + h[u]*np.log(u)
            y1 = y1 + h[u]
        
        #print(x0,x1)
        #print(y0,y1)
        
        if(y0 >0 and y1 >0): 
            k10=x0/y0
            k11=x1/y1
            
            #print(k10,k11)
        
            for u in range(1,t):
                z0 = z0 + h[u]*((np.log(u)-k10)**2)
                
            for u in range(t,len(h)):
                z1 = z1 + h[u]*((np.log(u)-k11)**2)
                    
            k20=z0/y0
            k21=z1/y1
            #print(z0,z1) 
            
            if (k20>0 and k21>0):
                k=P0*((1/2)*np.log(k20)-np.log(P0)) + P1*((1/2)*np.log(k21)-np.log(P1))
            if (k < mink and k!=-np.inf):
                mink=k
                mint=t
                       
    thresh=mint
        
    return(thresh)
  

def min_err_t_nakagami(h):     
    
    
    mint=0
    mink=np.inf
    
    for t in range(1,len(h)):
        P0=0
        P1=0
        x0=0
        y0=0
        z0=0
        x1=0
        y1=0
        z1=0
        k=np.inf
        
        for u in range(1,t):
            P0=P0+h[u]
            x0 = x0 + h[u]*np.log(u)
            y0 = y0 + h[u]
        
        for u in range(t,len(h)):
            P1=P1+h[u]
            x1 = x1 + h[u]*np.log(u)
            y1 = y1 + h[u]
        
        #print(x0,x1)
        #print(y0,y1)
        
        if(y0 >0 and y1 >0): 
            k10=x0/y0
            k11=x1/y1
            
            #print(k10,k11)
        
            for u in range(1,t):
                z0 = z0 + h[u]*((np.log(u)-k10)**2)
                
            for u in range(t,len(h)):
                z1 = z1 + h[u]*((np.log(u)-k11)**2)
                    
            k20=z0/y0
            k21=z1/y1
            
            def cost_func0(x):
                return abs(polygamma(1,x) - k20)
            
            def cost_func1(x):
                return abs(polygamma(1,x) - k21)  
            
            gamma0 = np.exp(2*k10)
            L0 = float(fsolve(cost_func0,2*k20))
            
            gamma1 = np.exp(2*k11)
            L1 = float(fsolve(cost_func1,2*k21))
            
            if(L0>0 and L1>0 and L0<75 and L1<75):
            
                k0 = (P0 * np.log(P0))
                
    
                for u in range(1,t):
                    try:
                        p0= ((2*math.gamma(2*L0))/(math.gamma(L0)**2))*(((gamma0**L0)*(u**(2*L0-1)))/((gamma0+u**2)**(2*L0)))
                    except OverflowError:
                        p0 = -np.inf
                    
                    k0 = k0+ h[u]*np.log(p0)
                    
                k1 = (P1 * np.log(P1))  
                for u in range(t,len(h)):
                    try:
                        p1= ((2*math.gamma(2*L1))/(math.gamma(L1)**2))*(((gamma1**L1)*(u**(2*L1-1)))/((gamma1+u**2)**(2*L1)))
                    except OverflowError:
                        p1 = -np.inf
                    k1 = k1 + h[u]*np.log(p1)
                    
                k=-(k0+k1)
                if (k < mink and k!=-np.inf):
                    mink=k
                    mint=t
                       
    thresh=mint
        
    return(thresh)
    
    

def min_err_t_wibull(h):     
    
    
    mint=0
    mink=np.inf
    
    for t in range(1,len(h)):
        P0=0
        P1=0
        x0=0
        y0=0
        z0=0
        x1=0
        y1=0
        z1=0
        k=np.inf
        
        for u in range(1,t):
            P0=P0+h[u]
            x0 = x0 + h[u]*np.log(u)
            y0 = y0 + h[u]
        
        for u in range(t,len(h)):
            P1=P1+h[u]
            x1 = x1 + h[u]*np.log(u)
            y1 = y1 + h[u]
        
        #print(x0,x1)
        #print(y0,y1)
        
        if(y0 >0 and y1 >0): 
            k10=x0/y0
            k11=x1/y1
            
            #print(k10,k11)
        
            for u in range(1,t):
                z0 = z0 + h[u]*((np.log(u)-k10)**2)
                
            for u in range(t,len(h)):
                z1 = z1 + h[u]*((np.log(u)-k11)**2)
                    
            k20=z0/y0
            k21=z1/y1
            #print(z0,z1) 

            lamb0 = np.exp(k10)
            n0 = np.sqrt(2*special.polygamma(1,1)/k20)
            
            lamb1 = np.exp(k11)
            n1 = np.sqrt(2*special.polygamma(1,1)/k21)
            
            k0 = (P0 * np.log(P0))
            
            for u in range(1,t):
                p0= n0*(lamb0**n0)*((u**(n0-1))/((lamb0**n0)+(u**n0))**2)
                k0 = k0+ h[u]*np.log(p0)
                
            k1 = (P1 * np.log(P1))  
            for u in range(t,len(h)):
                p1= n1*(lamb1**n1)*((u**(n1-1))/((lamb1**n1)+(u**n1))**2)
                k1 = k1 + h[u]*np.log(p1)
                
            k=-(k0+k1)
            if (k < mink and k!=-np.inf):
                mink=k
                mint=t
                       
    thresh=mint
        
    return(thresh)



VH, VV, truth = loadImageFromMatFile('MadridIowa_dual_1_corrige_1_', date = 0)
VH2, VV2, truth2 = loadImageFromMatFile('MadridIowa_dual_1_corrige_1_', date = 2)



se=strel('disk',50)

# dilatation
mask=morpho.dilation(truth2,se)

plt.figure()
plt.imshow(mask,cmap="gray")
plt.title('Mask Dilatation Rivière')
plt.plot()


VT_changement_mask = np.logical_xor(truth2, truth)
#VT_changement_mask = VT_changement_mask.astype(int)

plt.figure()
plt.imshow(VT_changement_mask, cmap='gray')
plt.title('Ground truth changement mask')
plt.plot()

plt.figure()
plt.imshow(truth, cmap='gray')
plt.title('Ground truth')
plt.plot()


#noise
    

noise1 = np.load('noisy_MadridIowa_dual_1_corrige_1_VH_AMPLITUDE_date_0.npy')
noise2 = np.load('noisy_MadridIowa_dual_1_corrige_1_VH_AMPLITUDE_date_2.npy')


#noise1 = skio.imread(path + 'noisy_MadridIowa_dual_1_corrige_1_VH_AMPLITUDE_date_0.npy')
#noise2 = skio.imread(path + 'noisy_MadridIowa_dual_1_corrige_1_VH_AMPLITUDE_date_2.png')


#rapport
image = noise1/noise2

plt.figure()
plt.imshow(image, cmap='gray')
plt.title('Image')
plt.plot()

#difference en log
#image = np.log(noise1)- np.log(noise2)
#image = abs(np.log(noise1/noise2))


#th = 200
#n_bins=int(np.floor(np.max(image[image<th])-np.min(image)+1))
#h = np.histogram(image, bins=n_bins)[0]

#image = np.zeros((512,512))
#image[0:255,:] = np.random.normal(50,15,(255,512))
#image[256:512,:] = np.random.normal(150,15,(256,512))


#image = np.zeros((512,512))
#image[0:255,:] = np.random.normal(38,9,(255,512))
#image[256:512,:] = np.random.normal(121,44,(256,512))

#image = np.zeros((512,512))
#image[0:255,:] = np.random.lognormal(0,np.sqrt(2),(255,512))
#image[256:512,:] = np.random.lognormal(1.5,np.sqrt(0.25),(256,512))

#image = np.zeros((512,512))
#image[0:255,:] = np.random.weibull(2,(255,512))
#image[256:512,:] = np.random.weibull(8,(256,512))

#weibull1 = stats.weibull_min.pdf(np.arange(0,4,0.01), 2)
#weibull2 = stats.weibull_min.pdf(np.arange(0,4,0.01), 8)
#plt.plot(np.arange(0,4,0.01), weibull1)
#plt.plot(np.arange(0,4,0.01), weibull2)

#lognorm1 = stats.lognorm.pdf(np.arange(0,4,0.01), 2,0)
#lognorm2 = stats.lognorm.pdf(np.arange(0,4,0.01), 0.25,0.1)
#plt.plot(np.arange(0,4,0.01), lognorm1)
#plt.plot(np.arange(0,4,0.01), lognorm2)
#plt.show()

max_range = np.quantile(image,0.9999)
n_bins= 100
h = np.histogram(image,range=(np.min(image),max_range), bins=n_bins)


t=min_err_t(h[0])
thresh_min= h[1][t]
min_binary = image > thresh_min

t_n=min_err_t_nakagami(h[0])
thresh_min_nakagami=h[1][t_n]
nakagami_binary = image > thresh_min_nakagami

t_l=min_err_t_lognorm(h[0])
thresh_min_lognorm=h[1][t_l]
lognorm_binary = image > thresh_min_lognorm

t_w=min_err_t_wibull(h[0])
thresh_min_weibull=h[1][t_w]
weibull_binary = image > thresh_min_weibull

print("thresh_min:",thresh_min,"\nthresh_min_nakagami:",thresh_min_nakagami,"\nthresh_min_lognor:",thresh_min_lognorm,"\nthresh_min_weibull:",thresh_min_weibull)

#image histogram with threhshold

plt.figure()
plt.hist(image.ravel(),range=(0,3*np.std(image)),bins=n_bins)
#plt.axvline(thresh_min, color='k', label='min error threhsold')
plt.axvline(thresh_min_nakagami, color='b', label='threhsold nakagami')
plt.axvline(thresh_min_lognorm, color='r', label='threhsold lognormal')
plt.axvline(thresh_min_weibull, color='g',linestyle="--", label='threhsold weibull')
plt.title('Histogram with the threhsold')
plt.legend()
plt.show()

plt.figure()
plt.imshow(min_binary, cmap='gray')
plt.title('Minimum error thresholding without mask')
plt.show()

plt.figure()
plt.imshow(nakagami_binary , cmap='gray')
plt.title('Nakagami threshold without mask')
plt.show()

plt.figure()
plt.imshow(lognorm_binary , cmap='gray')
plt.title('Lognormal threshold without mask')
plt.show()

plt.figure()
plt.imshow(weibull_binary, cmap='gray')
plt.title('Weibull threshold without mask')
plt.show()


# evaluate performance 

#Minimum error threhsold
print('Accuracy minimum error threhsold: ' + str(accuracy_score(VT_changement_mask.flatten(),min_binary.flatten())))
prec = precision_score(VT_changement_mask.flatten(),min_binary.flatten())
print('Precision minimum error threhsold: ' + str(prec))
rec = recall_score(VT_changement_mask.flatten(),min_binary.flatten())
print('Recall minimum error threhsold: ' + str(rec))
f1 = 2*prec*rec/(prec + rec)
print('F1 minimum error threhsold: ' + str(f1))
print('')
#Nakagami
print('Accuracy Nakagami threhsold: ' + str(accuracy_score(VT_changement_mask.flatten(),nakagami_binary.flatten())))
prec = precision_score(VT_changement_mask.flatten(),nakagami_binary.flatten())
print('Precision Nakagami threhsold: ' + str(prec))
rec = recall_score(VT_changement_mask.flatten(),nakagami_binary.flatten())
print('Recall Nakagami threhsold: ' + str(rec))
f1 = 2*prec*rec/(prec + rec)
print('F1 Nakagami threhsold: ' + str(f1))
print('')
#Lognorm
print('Accuracy Lognormal threhsold: ' + str(accuracy_score(VT_changement_mask.flatten(),lognorm_binary.flatten())))
prec = precision_score(VT_changement_mask.flatten(),lognorm_binary.flatten())
print('Precision Lognormal threhsold: ' + str(prec))
rec = recall_score(VT_changement_mask.flatten(),lognorm_binary.flatten())
print('Recall Lognormal threhsold: ' + str(rec))
f1 = 2*prec*rec/(prec + rec)
print('F1 Lognormal threhsold: ' + str(f1))
print('')
#Weibull
print('Accuracy Weibull threhsold: ' + str(accuracy_score(VT_changement_mask.flatten(),weibull_binary.flatten())))
prec = precision_score(VT_changement_mask.flatten(),weibull_binary.flatten())
print('Precision Weibull threhsold: ' + str(prec))
rec = recall_score(VT_changement_mask.flatten(),weibull_binary.flatten())
print('Recall Weibull threhsold: ' + str(rec))
f1 = 2*prec*rec/(prec + rec)
print('F1 Weibull threhsold: ' + str(f1))
print('')




# dilated mask

plt.figure()
plt.imshow(mask*min_binary, cmap='gray')
plt.title('Minimum error thresholding with dilated mask')
plt.show()

plt.figure()
plt.imshow(mask*nakagami_binary, cmap='gray')
plt.title('Nakagami thresholding with dilated mask')
plt.show()

plt.figure()
plt.imshow(mask*lognorm_binary, cmap='gray')
plt.title('Lognormal thresholding with dilated mask')
plt.show()

plt.figure()
plt.imshow(mask*weibull_binary, cmap='gray')
plt.title('Weibull thresholding with dilated mask')
plt.show()



# evaluate performance mask

#Minimum error threhsold
print('Accuracy minimum error threhsold: ' + str(accuracy_score(VT_changement_mask.flatten(),(mask*min_binary).flatten())))
prec = precision_score(VT_changement_mask.flatten(),(mask*min_binary).flatten())
print('Precision minimum error threhsold: ' + str(prec))
rec = recall_score(VT_changement_mask.flatten(),(mask*min_binary).flatten())
print('Recall minimum error threhsold: ' + str(rec))
f1 = 2*prec*rec/(prec + rec)
print('F1 minimum error threhsold: ' + str(f1))
print('')
#Nakagami
print('Accuracy Nakagami threhsold: ' + str(accuracy_score(VT_changement_mask.flatten(),(mask*nakagami_binary).flatten())))
prec = precision_score(VT_changement_mask.flatten(),(mask*nakagami_binary).flatten())
print('Precision Nakagami threhsold: ' + str(prec))
rec = recall_score(VT_changement_mask.flatten(),(mask*nakagami_binary).flatten())
print('Recall Nakagami threhsold: ' + str(rec))
f1 = 2*prec*rec/(prec + rec)
print('F1 Nakagami threhsold: ' + str(f1))
print('')
#Lognorm
print('Accuracy Lognormal threhsold: ' + str(accuracy_score(VT_changement_mask.flatten(),(mask*lognorm_binary).flatten())))
prec = precision_score(VT_changement_mask.flatten(),(mask*lognorm_binary).flatten())
print('Precision Lognormal threhsold: ' + str(prec))
rec = recall_score(VT_changement_mask.flatten(),(mask*lognorm_binary).flatten())
print('Recall Lognormal threhsold: ' + str(rec))
f1 = 2*prec*rec/(prec + rec)
print('F1 Lognormal threhsold: ' + str(f1))
print('')
#Weibull
print('Accuracy Weibull threhsold: ' + str(accuracy_score(VT_changement_mask.flatten(),(mask*weibull_binary).flatten())))
prec = precision_score(VT_changement_mask.flatten(),(mask*weibull_binary).flatten())
print('Precision Weibull threhsold: ' + str(prec))
rec = recall_score(VT_changement_mask.flatten(),(mask*weibull_binary).flatten())
print('Recall Weibull threhsold: ' + str(rec))
f1 = 2*prec*rec/(prec + rec)
print('F1 Weibull threhsold: ' + str(f1))
print('')



#############################################################################################

#denoise

denoise1 = np.load('denoised_MadridIowa_dual_1_corrige_1_VH_AMPLITUDE_date_0.npy')
denoise2 = np.load('denoised_MadridIowa_dual_1_corrige_1_VH_AMPLITUDE_date_2.npy')

#rapport
image = denoise1/denoise2

#difference
#image = denoise1 - denoise2


max_range = np.quantile(image,0.9999)
n_bins= 1000
h = np.histogram(image,range=(np.min(image),max_range), bins=n_bins)


t=min_err_t(h[0])
thresh_min= h[1][t]
min_binary = image > thresh_min

t_n=min_err_t_nakagami(h[0])
thresh_min_nakagami=h[1][t_n]
nakagami_binary = image > thresh_min_nakagami

t_l=min_err_t_lognorm(h[0])
thresh_min_lognorm=h[1][t_l]
lognorm_binary = image > thresh_min_lognorm

t_w=min_err_t_wibull(h[0])
thresh_min_weibull=h[1][t_w]
weibull_binary = image > thresh_min_weibull

print("thresh_min:",thresh_min,"\nthresh_min_nakagami:",thresh_min_nakagami,"\nthresh_min_lognor:",thresh_min_lognorm,"\nthresh_min_weibull:",thresh_min_weibull)


#image histogram with threhshold

plt.figure()
plt.hist(image.ravel(),range=(np.min(image),3*np.std(image)), bins=n_bins)
plt.axvline(thresh_min, color='k', label='min error threhsold')
plt.axvline(thresh_min_nakagami, color='b', label='threhsold nakagami')
plt.axvline(thresh_min_lognorm, color='r', label='threhsold lognormal')
plt.axvline(thresh_min_weibull, color='g',linestyle="-", label='threhsold weibull')
plt.title('Histogram with the threhsold')
plt.legend()
plt.show()

plt.figure()
plt.imshow(min_binary, cmap='gray')
plt.title('Minimum error thresholding without mask')
plt.show()

plt.figure()
plt.imshow(nakagami_binary , cmap='gray')
plt.title('Nakagami threshold without mask')
plt.show()

plt.figure()
plt.imshow(lognorm_binary , cmap='gray')
plt.title('Lognormal threshold without mask')
plt.show()

plt.figure()
plt.imshow(weibull_binary, cmap='gray')
plt.title('Weibull threshold without mask')
plt.show()



# evaluate performance 

#Minimum error threhsold
print('Accuracy minimum error threhsold: ' + str(accuracy_score(VT_changement_mask.flatten(),min_binary.flatten())))
prec = precision_score(VT_changement_mask.flatten(),min_binary.flatten())
print('Precision minimum error threhsold: ' + str(prec))
rec = recall_score(VT_changement_mask.flatten(),min_binary.flatten())
print('Recall minimum error threhsold: ' + str(rec))
f1 = 2*prec*rec/(prec + rec)
print('F1 minimum error threhsold: ' + str(f1))
print('')
#Nakagami
print('Accuracy Nakagami threhsold: ' + str(accuracy_score(VT_changement_mask.flatten(),nakagami_binary.flatten())))
prec = precision_score(VT_changement_mask.flatten(),nakagami_binary.flatten())
print('Precision Nakagami threhsold: ' + str(prec))
rec = recall_score(VT_changement_mask.flatten(),nakagami_binary.flatten())
print('Recall Nakagami threhsold: ' + str(rec))
f1 = 2*prec*rec/(prec + rec)
print('F1 Nakagami threhsold: ' + str(f1))
print('')
#Lognorm
print('Accuracy Lognormal threhsold: ' + str(accuracy_score(VT_changement_mask.flatten(),lognorm_binary.flatten())))
prec = precision_score(VT_changement_mask.flatten(),lognorm_binary.flatten())
print('Precision Lognormal threhsold: ' + str(prec))
rec = recall_score(VT_changement_mask.flatten(),lognorm_binary.flatten())
print('Recall Lognormal threhsold: ' + str(rec))
f1 = 2*prec*rec/(prec + rec)
print('F1 Lognormal threhsold: ' + str(f1))
print('')
#Weibull
print('Accuracy Weibull threhsold: ' + str(accuracy_score(VT_changement_mask.flatten(),weibull_binary.flatten())))
prec = precision_score(VT_changement_mask.flatten(),weibull_binary.flatten())
print('Precision Weibull threhsold: ' + str(prec))
rec = recall_score(VT_changement_mask.flatten(),weibull_binary.flatten())
print('Recall Weibull threhsold: ' + str(rec))
f1 = 2*prec*rec/(prec + rec)
print('F1 Weibull threhsold: ' + str(f1))
print('')




# dilated mask

plt.figure()
plt.imshow(mask*min_binary, cmap='gray')
plt.title('Minimum error thresholding with dilated mask')
plt.show()

plt.figure()
plt.imshow(mask*nakagami_binary, cmap='gray')
plt.title('Nakagami thresholding with dilated mask')
plt.show()

plt.figure()
plt.imshow(mask*lognorm_binary, cmap='gray')
plt.title('Lognormal thresholding with dilated mask')
plt.show()

plt.figure()
plt.imshow(mask*weibull_binary, cmap='gray')
plt.title('Weibull thresholding with dilated mask')
plt.show()


# evaluate performance mask

#Minimum error threhsold
print('Accuracy minimum error threhsold: ' + str(accuracy_score(VT_changement_mask.flatten(),(mask*min_binary).flatten())))
prec = precision_score(VT_changement_mask.flatten(),(mask*min_binary).flatten())
print('Precision minimum error threhsold: ' + str(prec))
rec = recall_score(VT_changement_mask.flatten(),(mask*min_binary).flatten())
print('Recall minimum error threhsold: ' + str(rec))
f1 = 2*prec*rec/(prec + rec)
print('F1 minimum error threhsold: ' + str(f1))
print('')
#Nakagami
print('Accuracy Nakagami threhsold: ' + str(accuracy_score(VT_changement_mask.flatten(),(mask*nakagami_binary).flatten())))
prec = precision_score(VT_changement_mask.flatten(),(mask*nakagami_binary).flatten())
print('Precision Nakagami threhsold: ' + str(prec))
rec = recall_score(VT_changement_mask.flatten(),(mask*nakagami_binary).flatten())
print('Recall Nakagami threhsold: ' + str(rec))
f1 = 2*prec*rec/(prec + rec)
print('F1 Nakagami threhsold: ' + str(f1))
print('')
#Lognorm
print('Accuracy Lognormal threhsold: ' + str(accuracy_score(VT_changement_mask.flatten(),(mask*lognorm_binary).flatten())))
prec = precision_score(VT_changement_mask.flatten(),(mask*lognorm_binary).flatten())
print('Precision Lognormal threhsold: ' + str(prec))
rec = recall_score(VT_changement_mask.flatten(),(mask*lognorm_binary).flatten())
print('Recall Lognormal threhsold: ' + str(rec))
f1 = 2*prec*rec/(prec + rec)
print('F1 Lognormal threhsold: ' + str(f1))
print('')
#Weibull
print('Accuracy Weibull threhsold: ' + str(accuracy_score(VT_changement_mask.flatten(),(mask*weibull_binary).flatten())))
prec = precision_score(VT_changement_mask.flatten(),(mask*weibull_binary).flatten())
print('Precision Weibull threhsold: ' + str(prec))
rec = recall_score(VT_changement_mask.flatten(),(mask*weibull_binary).flatten())
print('Recall Weibull threhsold: ' + str(rec))
f1 = 2*prec*rec/(prec + rec)
print('F1 Weibull threhsold: ' + str(f1))
print('')