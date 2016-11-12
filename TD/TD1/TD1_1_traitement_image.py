# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 11:26:44 2015

@author: Aurelien Plyer
"""

# importation des modules qu'on va utiliser :
import cv2                # OpenCV
import pylab as pl        # pylab pour l'affichage
import numpy as np        # numpy pour les calculs sur matrices
from scipy import ndimage # scipy pour les convolution


#%%
# Chargement des données
data_dir = '/home/viki/data_td/perception/TD1/matching'
images_name = data_dir+'/features2d_%d.jpg'
patterns_name = data_dir+'/features2d_%s.jpg'
patterns_type = ['java', 'opengl', 'qt4']
images = [cv2.imread(images_name%i,cv2.IMREAD_GRAYSCALE) for i in range(1,8)]
patterns = [cv2.imread(patterns_name%i,cv2.IMREAD_GRAYSCALE) for i in patterns_type]


#%%
#
# affichage de toutes les images collées les unes à coté des autres :
#
pl.figure()
pl.imshow(np.concatenate(images, axis = 1),'gray')

#%%
#
# Question : comment les coller verticalement plutot?
#

pl.figure()
# Reponse :


#%%
#
# Question : comment faire un affichage en grille 2 x 4 plutot ?
#
pl.figure()

# Reponse :

pl.imshow(grid,'gray')


#%%
# affichage de tous les patterns collées les un à coté des autres :
pl.figure()
pl.imshow(np.concatenate(patterns, axis = 1),'gray')

###############################################
# Question : pourquoi cela ne fonctionne pas? #
###############################################
# Réponse :
###############################################

#%%
#
###############################################
# Question : calculer le vecteur des resolution d'images de la séquence de patterns
#
###############################################
# Reponse
patterns_size = []

print(patterns_size)

#%%
#
###############################################
# Question : calculer une liste patterns_crop de pattern de resolution (480, 320)
# et calculer la liste de ses resolution
###############################################
# Reponse :
patterns_crop = []
patterns_crop_size = []

print(patterns_crop_size)

pl.figure()
pl.imshow(np.concatenate(patterns_crop, axis = 1),'gray')

#%%
#
###############################################
# Question : calculer une liste des patterns_crop renversé (tete en bas) en travaillant
# sur les indices
###############################################
#Reponse :
patterns_flip = []

figure()
imshow(np.concatenate(patterns_flip, axis = 1),'gray')


#%%
#################################
# Un peut de traitement d'image #
#################################
#
# Pour les fonction de convolution on utilise les outils
# du module ndimage de scipy
#

I = images[0]

#%%
# les convolutions
# pour avoir l'aide taper dans le terminal ipython : %ndimage.convolve
# (touche q pour quiter l'aide)
#

print('les convolutions')

#%%
#
# construiser un noyau de convolution (15,15) remplis de 1
# et convoluer I avec en utilisant ndimage.convolve
#

ker =
Ic  =

figure()
imshow(np.concatenate([I,Ic], axis = 1), 'gray')

#############################################################
# Question : Que ce passe-t-il? est-ce le resultat attendu? #
#############################################################
# Reponse :
#
#############################################################

#%%
#  regardons quel est le type de donnée des pixels de I
#

print('type de l\'image I : '+ str(type(I)))
print('type du pixel de I : '+ str(type(I[0,0])))

#############################################################
# Question : conbien de niveau contient un uint8 ?
#############################################################
# Reponse :
#
#############################################################

#%%
# passage en calcul flotant
#

I = I.astype(np.float32)

print('type de l\'image I : '+str(type(I)))
print('type du pixel de I : '+str(type(I[0,0])))

#%%
# on re-tente une convolution
#

ker =
Ic  =

figure()
imshow(np.concatenate([I,Ic], axis = 1), 'gray')

#############################################################
# Question : Que ce passe-t-il? pourquoi Ic est noir?        #
#############################################################
# Reponse :
#
#############################################################


#%%
#
#  calcul des valeures min-max et moyenne d'une image
#

def print_minmax(I):
    Imin = np.min(I)
    Imax = np.max(I)
    Imean = np.mean(I)
    print ('min : %f  / max : %f  / mean : %f'%(Imin, Imax, Imean))

print_minmax(I)
print_minmax(Ic)

##############################################################
# Question : comment garder Ic dans la même dynamique que I? #
##############################################################
# Reponse :
#
##############################################################

#%%
#########################################################################
# Question : normaliser le noyau de manière à ce que sa moyenne fasse 1
########################################################################
# Repoonse :
ker =
#########################################################################

Ic = ndimage.convolve(I,ker)

figure()
imshow(np.concatenate([I,Ic], axis = 1), 'gray')


print_minmax(I)
print_minmax(Ic)

# c'est mieux non ?

#%%
###############################
# Les opérateurs de gradients #
###############################

#################################################################
# Question : quels sont les différents operateurs de gradients ?
#################################################################
# Reponse :
#
#################################################################
# Question : Implementer les operateurs de gradient centrés en X et Y
#################################################################
# Reponse :


#################################################################


Ix = ndimage.convolve(I,gx)
Iy = ndimage.convolve(I,gy)

figure()
imshow(np.concatenate([Ix,Iy],axis = 1), 'gray')

#%%
# calcul de la norme du gradient
#

N = np.sqrt(Ix**2 + Iy**2)
figure()
imshow(N)

print_minmax(N)

#%%
# calcul de l'histogram d'une image,
# [INFO la methode 'ravel()' permet de transformer une matrice en un vecteur
# monodimensionnel]
#

hist, bins = np.histogram(N.ravel(), range(256))
figure()
plot(bins[:-1],hist)

#%%
# calculer une image binaire des norme de gradient
#

###########################################################################
# Question : d'apres l'histogramme à quel valeur peut-on mettre le seuil?
############################################################################
# Reponse :
#
seuil =

############################################################################

contour = np.zeros(I.shape)
contour[N > seuil] = 1

figure()
imshow(contour, 'gray')

#%%
#
# effectuer un seuillage par histérésis avec deux seuils : seuils bas
# et seuil haut avec un masque de 8 connexité (np.ones([3,3])
#

seuil_bas = 10
seuil_haut = 30

#############################################################################
# Reponse
############################################################################


############################################################################

pl.figure()
pl.imshow(result)
pl.figure()
pl.imshow(haut)
pl.figure()
pl.imshow(bas)


#%%
#
# Gradient en utilisant le Canny d'opencv
#

edges = cv2.Canny(I,50,200,apertureSize = 3)

pl.figure()
pl.imshow(edges)

#%%
#
# LA SUITE EST OPTIONNELLE
#

print('partie optionnelle')

#%%
#
# une problematique fréquente est d'extraire des lignes dans les image
# a partir des gradients images, pour ca au lieu de réécrir des operateurs
# de gradient on peut utiliser l'opérateur de Canny et une transformé de Hough
#

I = images[0]
img = I.copy()
img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
edges = cv2.Canny(I,50,200,apertureSize = 3)

figure()
imshow(edges, 'gray')
lines = cv2.HoughLines(edges,1,np.pi/180, 200)
for rho,theta in lines.reshape(-1,2):
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

figure()
imshow(img)



