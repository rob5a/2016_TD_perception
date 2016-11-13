# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 13:37:47 2015

@author: Aurélien Plyer
"""
print 'importation des modules'
import cv2                # OpenCV
import pylab as pl        # pylab pour l'affichage
import numpy as np        # numpy pour les calculs sur matrices
from scipy import ndimage # scipy pour les convolution
from mpl_toolkits.mplot3d import Axes3D
from pylab import plot
import glob

#%%
#
# On récupère ce qu'on a fait au TD précédent
#

import sys
sys.path.append('/home/viki/Documents/perception/TD/TD2')
from td1 import Matcheur, drawMatches

Matcheur.InitStatic()

#%%

print 'Chargement des données '

sequence = 'drone_stereo'
data_loc = '/home/viki//data_td/perception/TD2/calcul_pose/'
dataset = data_loc+sequence+'/%05d_%02d.png'
geometry = np.load(data_loc+sequence+'/geometry.npz')
T01 = geometry['T01']
K0 = geometry['K0']
K1 = geometry['K1']
baseline = -T01[0]

frames_number = len(glob.glob(data_loc+sequence+'/*_00.png'))

V0=[cv2.imread(dataset%(i+1,0), cv2.IMREAD_GRAYSCALE) for i in range(frames_number)]
V1=[cv2.imread(dataset%(i+1,1), cv2.IMREAD_GRAYSCALE) for i in range(frames_number)]

def play():
    '''
       juste pour visualiser la séquence
    '''
    cv2.namedWindow('video',cv2.WINDOW_NORMAL)
    for I0, I1 in zip(V0,V1):
        cv2.imshow('video', np.concatenate([I0,I1],axis = 1))
        cv2.waitKey(100)

#%%
#
# relecture de la video
#
play()
#%%
# pour fermer la fenetre
cv2.destroyWindow('video')
#%%
#
# Construction d'un matcheur adapté a du traitement video
#
matcheur = Matcheur(detecteur ='fast', descripteur ='orb')

#%%
# triangulation des points stereo
#
def matchAndTriangulePoints(matcheur, I0, I1, K0, K1, baseline):
    _, _ = matcheur.match(I0,I1)
    is_valid, _, _, _ = matcheur.affine_match('stereo')
    pts0, pts1, kp0, kp1, desc0, desc1 = matcheur.get_valid_kp()
    #
    # Calcul du nuage de points à partir de la disparite
    #
    disparity = np.expand_dims(pts1[:,0] - pts0[:,0],axis = 1)
    u = np.expand_dims(pts0[:,0], axis = 1)
    v = np.expand_dims(pts0[:,1], axis = 1)
    #########################################################################
    # Question : à partir de la mesure de disparité et des informations
    #            de calibration veuillez calculer les coordonées 3D des
    #            points mis en correspondance dans un vecteur pts3D (N,3)
    ########################################################################
    # Reponse :


    ##########################################################################
    return pts3D, pts0, pts1, kp0, kp1, desc0, desc1


#%%
#
#

def play3D(ax):
    def draw_triedre(ax):
        ax.plot([0,0.5],[0,0],[0,0], c=(1.,0.,0.),lw = 3)
        ax.plot([0,0],[0,0.5],[0,0], c=(0.,1.,0.),lw = 3)
        ax.plot([0,0],[0,0],[0,0.5], c=(0.,0.,1.),lw = 3)
    for i in range(len(V0)):
        pts3D, pts0, pts1, kp0, kp1, desc0, desc1 =  matchAndTriangulePoints(matcheur, V0[i], V1[i], K0, K1, baseline)
        ax.clear()
        # pour passer du repere vision par ordinateur a un repere plus
        # commun il suffit de faire la permutation d'axe suivante
        # X = X
        # Y = Z
        # Z = -Y
        ax.set_xlim3d(-10, 10)
        ax.set_ylim3d(-10,10)
        ax.set_zlim3d(-1,10)
        figure(1)
        ax.scatter(pts3D[::50,0],pts3D[::50,2],-pts3D[::50,1])
        draw_triedre(ax)
        pause(0.05)
        figure(2)
        imshow(V0[i],'gray')
        pause(0.05)

################################
#%%
################################
fig = figure(1)
ax = fig.gca(projection='3d')
figure(2)
#%%
#
# pour (re-)jouer l'affichage 3D
#

play3D(ax)

#%%
#
# Nous avons réussi à obtennir à partir d'une paire d'image un nuage de points
# 3D, le probleme qui se pose maintenant est celui du calcul de pose, pour
# cela on va utiliser 2 mises en correspondance :
#   - C1 = correspondance gauche-droite 'stereo' à t0
#   - C2 = correspondance gauche-gauche 'fundamental' entre t0 et t1
# pour que cela fonctionne, il faut que C2 soit calculé en utilisant les keypoints
# ayant été valide lors du calcul stéréo à t0
#
# avec ces deux mise en correspondance, on va pouvoir calculer le deplacement 3D
# du capteur dans le temps
#

temps0 = 0
temps1 = 1

##########################################################
# Question : calculer la correpondance stereo à temps 0
##########################################################
# Reponse :


##########################################################

#%%
###########################################################
# Question : Calculez les keypoints gauche à temps1 (kp0_1)
# et les mettre en correspondance avec kp0_0
# vous pouvez accèder directement aux membres 'detector', 'descriptor'
# et 'matcheur' de la classe matcheur afin de simplifier le code
##########################################################
# Reponse :

##########################################################

img = drawMatches(V0[temps0], kp0_0, V0[temps1], kp0_1, match_0_1)
figure()
imshow(img)

#%%
# Extraction des appariements valide au sens de la géométrie épipolaire
# (matrice fondamentale)
#
pts0_0, pts0_1 = Matcheur.getPointsFromMatch(match_0_1, kp0_0, kp0_1)

####################################################################
# Question : filtrer l'appariement en utilisant un estimation par RANSAC
#            de la matrice fondamentale et donnez le nombre d'inlier/outlier
####################################################################
# Reponse :


########################################################################

print 'inlier  : ', inlier
print 'outlier : ', outlier

#%%
#
# Affichage des match valide au sens de la fondamentale estimé
#

valid = [ m for idx, m in enumerate(match_0_1) if mask[idx] == 1]
pts0_0 = np.array([kp0_0[m.queryIdx].pt for m in valid])
pts0_1 = np.array([kp0_1[m.trainIdx].pt for m in valid])
pts3D_valid = np.array([pts3D_0[m.queryIdx,:]for m in valid])
img = drawMatches(V0[temps0],kp0_0,V0[temps1],kp0_1,valid)
figure()
imshow(img)

#%%
######################################################################
#  Calcul de pose temps0 -> temps1
#  nous allons ici tester des différentes fonction de calcul de pose.
#  lorsqu'on connais la position 3D des points (pts3D), ce problème est
#  appelé calcul de pose absolue et se résouds par la famille d'algorithmes
#  appelé PnP
#######################################################################

N = pts0_1.shape[0]
object_pts = pts3D_valid.reshape([N,1,3]).astype(np.float32)
image_pts  = pts0_1.reshape([N,1,2]).astype(np.float32)

######################################################################
# Question : utilisez les fonction solvePnPRansac  et solvePnP avec les
#            options cv2.CV_EPNP et cv2.ITERATIVE
#            pour calculer la pose de la cameras au temps1
#######################################################################
# Reponse :
#%%
print '========================================'
#
print 'resultat PnP : '
print 'translation : (%02.02f,%02.02f,%02.02f) '%(t_0_1[0],t_0_1[1],t_0_1[2])
print 'rotation    : (%02.02f,%02.02f,%02.02f) '%(R_0_1[0], R_0_1[1], R_0_1[2])
print '========================================'
#%%
#
print 'resultat PnP iterative  : '
print 'translation : (%02.02f,%02.02f,%02.02f) '%(t_0_1[0],t_0_1[1],t_0_1[2])
print 'rotation    : (%02.02f,%02.02f,%02.02f) '%(R_0_1[0], R_0_1[1], R_0_1[2])
print '========================================'
#%%
#
print 'resultat RANSAC : '
print 'translation : (%02.02f,%02.02f,%02.02f) '%(t_0_1[0],t_0_1[1],t_0_1[2])
print 'rotation    : (%02.02f,%02.02f,%02.02f) '%(R_0_1[0], R_0_1[1], R_0_1[2])
print '========================================'
#%%
# time EPNP : 895 us
# time iterative : 11.3 ms
# time ransac : 3.66ms
#
###########################################################################
# Question : Quel est la solution qui vous semble la plus vraissemblable
#            (sachant que les mesures sont données en mètre et radian)?
#            Pour quel raison avons nous ce resultat selon vous?
############################################################################
# Reponse :
#
############################################################################


#%%
#
# Pour obtennir la transformation des points du temps0 (objet) dans le
# repère de la caméra au temps1 il suffit alors d'utiliser la fonction
# de conversion cv2.Rodrigues
#

R_0_to_1, _ = cv2.Rodrigues(R_0_1)
T_0_to_1 = t_0_1

############################################################################
# Question : matchez et triangulé le nuage de points au temps1
############################################################################
# Reponse :


#############################################################################


##############################################################################
# Question : Calcul de la position du nuage_0 dans le repere_1
# pts3D_0_in_1 = R_0_to_1 * pts3D_0 + T_0_to_1
##############################################################################
# Reponse :

##############################################################################
print pts3D_0_in_1.shape

#%%
#
# visualisons le résultat
# ce qui nous interessera ici sera plutot de nous interesser à la transformation
# du repere 1 vers le repere 0 d'origine qui sera fixé comme reference dans
# le petit bout de sequence video
#

#########
# Tout d'abord écrivons une fonction permettant d'aficher un trièdre d'une
# pause
#########
def draw_triedre(ax,R = np.eye(3),T = np.zeros([3,1]), size = 0.5):
    orig = np.zeros([3,1])
    x = np.array([[size],[0.],[0.]])
    y = np.array([[0.],[size],[0.]])
    z = np.array([[0.],[0.],[size]])
    ###########################################################################
    # Question : Calculer la position dans le repere monde des points orig,x,y
    #            et z charactérisant le triedre de la pose calculé
    ###########################################################################
    # Reponse :



    ###########################################################################
    ax.plot([orig[0,0],x[0,0]],[orig[2,0],x[2,0]],[-orig[1,0],-x[1,0]], c=(1.,0.,0.),lw = 3)
    ax.plot([orig[0,0],y[0,0]],[orig[2,0],y[2,0]],[-orig[1,0],-y[1,0]], c=(0.,1.,0.),lw = 3)
    ax.plot([orig[0,0],z[0,0]],[orig[2,0],z[2,0]],[-orig[1,0],-z[1,0]], c=(0.,0.,1.),lw = 3)

##############################
#%%
##############################
fig = figure(1)
ax = fig.gca(projection='3d')
figure(2)
############################

#%%
#
# Création de la fonction d'odométrie calculant et affichant son résultat
#

def playOdometer3D(num_images):
    temps0 = 0
    ax.clear()
    ax.set_xlim3d(-10, 10)
    ax.set_ylim3d(-10, 10)
    ax.set_zlim3d(-1,5)
    R = [np.eye(3)]
    T = [np.zeros([3,1])]
    ###########################################################################
    # Question : calculez du nuage de points initial (temps0)
    ###########################################################################
    # Reponse :

    ###########################################################################
    ax.scatter(pts3D_0[:,0], pts3D_0[:,2],-pts3D_0[:,1], s=40, c=(1.,0,0))
    for i in range(1,num_images):
        temps1 = i
        #######################################################################
        # Question : assemblez les fonctions précédement vue (suivit temporel +
        #           calcul de pose)
        #######################################################################
        # Reponse :
        ##########################
        # association temporel :
        ##########################


        #########################################
        # filtre fondamental des associations :
        #########################################


        ####################
        # Calcul de pose :
        ####################



        ######################################
        # Triangulation des points temps 1 :
        ######################################



        ########################################################################
        # Question : calculez la position du nuage de points pts3D_1 dans le
        #            repere de la camera a temps0
        ########################################################################
        # Reponse :
        pts3D_1_in_0 =
        ########################################################################
        R.append(R_1_to_0)
        T.append(T_1_to_0)
        ax.clear()
        ax.set_xlim3d(-10, 10)
        ax.set_ylim3d(-10,10)
        ax.set_zlim3d(-1,5)
        ax.scatter(pts3D_0[::10,0],pts3D_0[::10,2],-pts3D_0[::10,1], s=40, c=(0.,0,1.))
        ax.scatter(pts3D_1_in_0[::50,0],pts3D_1_in_0[::50,2],-pts3D_1_in_0[::50,1], s=20, c=(1.,0,0))
        for r,t in zip(R,T):
            draw_triedre(ax,r, t, size = 0.5)
        print 'temps %02d : (%02.02f,%02.02f,%02.02f) inliers ratio %f'%(i,T_0_to_1[0],T_0_to_1[1],T_0_to_1[2], inlier_ratio)
        figure(1)
        pause(0.05)
        figure(2)
        imshow(img,'gray')
        pause(0.05)


#%%
############################################################
# lancement de l'odométrie sur les 20 premières images
############################################################
playOdometer3D(20)


####################################################################
# Question : Qu'observez vous? Comment améliorer le calcul de pose #
####################################################################
# Reponse :
#
############################################################

