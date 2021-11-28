#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 22 May 2019

@author: M Roux
"""


import math
import numpy as np
import matplotlib.pyplot as plt

from scipy import ndimage
from scipy import signal

from skimage import io

##############################################

import mrlab as mr

##############################################"

############## le close('all') de Matlab
plt.close('all')
################################"


alpha=0.1
seuilnorme=20
ima=io.imread(r'C:\Users\jutogashi\Documents\Telecom_2A\IMA\IMA201\TP3\image\pyra-gauss.tif')



gradx=mr.dericheGradX(mr.dericheSmoothY(ima,alpha),alpha)
grady=mr.dericheGradY(mr.dericheSmoothX(ima,alpha),alpha)  

   
plt.figure('Image originale')
plt.imshow(ima, cmap='gray')

plt.figure('Gradient horizontal')
plt.imshow(gradx, cmap='gray')

plt.figure('Gradient vertical')
plt.imshow(grady, cmap='gray')

norme=np.sqrt(gradx*gradx+grady*grady)

plt.figure('Norme du gradient')
plt.imshow(norme, cmap='gray')



#io.imsave('norme.tif',np.uint8(norme))

nl,nc=gradx.shape
direction=np.arctan2(np.ravel(grady),np.ravel(gradx));

direction=np.reshape(direction,(nl, -1))
direction=255*direction/2/math.pi;

plt.figure('Direction du gradient')
plt.imshow(direction)

io.imsave('direction.tif',np.uint8(direction))

contours=np.uint8(mr.maximaDirectionGradient(gradx,grady))

plt.figure('Contours')
plt.imshow(255*contours)


valcontours=(norme>seuilnorme)*contours
      
plt.figure('Contours normés')
plt.axis("off")
plt.imshow(255*valcontours)

#io.imsave('contours.tif',np.uint8(255*valcontours))

  
