# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 10:26:49 2015

@author: Aurélien Plyer
"""

#
#
# Partie plus traitement de données, objectif manipuler stereo et flot
# et voir les couts de calculs et les differents propriete des algos
#


print 'importation des modules'
import cv2                # OpenCV
import pylab as pl        # pylab pour l'affichage
import numpy as np        # numpy pour les calculs sur matrices
from scipy import ndimage # scipy pour les convolution
from mpl_toolkits.mplot3d import Axes3D
import time

#%%
#
#  Ouverture des données
#
paire1, paire2, paire3, paire4, paire5 = pickle.load(open('/home/viki/data_td/perception/TD3/td3_stereo.p','r'))


#%%
#
# Calcul de disparite
#
# choix de la paire de travail

def testStereoAlgo(I0,I1, winSize, numDisp):
    '''
        Fonction calculant et affichant la carte de disparité avec 4 algorighmes:
        BM : algorithme de bloque matching stereo simple
        SGBM1 : algorithme sgbm avec une faible régularisaion
        SGBM2 : algorithme sgbm avec une régularisation moyenne
        SGMB3 : algorithme sgbm avec une forte régularisation

        winSize = taille de la fenêtre de corrélation
        numDisp = nombre d'hypothèse de disparités
    '''
    print(numDisp)
    bm = cv2.StereoBM(cv2.STEREO_BM_BASIC_PRESET,numDisp, winSize)
    disp1 = bm.compute(I0,I1).astype(np.float32) /16.
    sgbm = cv2.StereoSGBM(0,numDisp, winSize, P1 = 0, P2 = 5)
    disp2 = sgbm.compute(I0,I1).astype(np.float32) /16.
    sgbm2 = cv2.StereoSGBM(0,numDisp, winSize, P1 = 10, P2 = 500, speckleWindowSize = 51, speckleRange= 1 )
    disp3 = sgbm2.compute(I0,I1).astype(np.float32) /16.
    sgbm3 = cv2.StereoSGBM(0,numDisp, winSize, P1 = 5000, P2 = 10000, speckleWindowSize = 51, speckleRange= 1)
    disp4 = sgbm3.compute(I0,I1).astype(np.float32) /16.
    figure()
    disp = np.concatenate([np.concatenate([disp1, disp3]),np.concatenate([disp2,disp4])],axis = 1)
    imshow(disp, vmin= 0, vmax = numDisp)
    colorbar()
    title('resultat [[BM ,SGBM1],[SGBM 2, SGBM3]]')

#%%
###############################################################################
# Question : étudiez le comportement des algorithmes de stéréo en faisant
#            varier le paramẽtre de fenetre de correlation dans [5,7,9,19,25]
#            le paramètre d'hypothèse de disparité dans [16,32,64,128]
#            sur les 4 paires [paire1, paire2, paire3, paire4]
#            vous sauvegarderez les résultats avec :
#            savefig('/data/tp3_stereo_paire%d_disp03d_rad%02d.png'%(n_paire,disp,radius),bbox_inches='tight')
###############################################################################
# Reponse :


###############################################################################

###############################################################################
# Question : Interpétez les résultats, quel est pour vous le meilleur algorithm
#            pour chacune des paires? quel est le paramètrage en rayon/disparite
#            qui vous semble le mieux?
#            quel est l'effet de la fenetre de corrélation sur l'échelle de la
#            carte de disparité?
###############################################################################
# Réponse :
#
###############################################################################


