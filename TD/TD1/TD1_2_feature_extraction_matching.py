# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 12:27:14 2015

@author: Aurelien Plyer
"""

# importation des modules qu'on va utiliser :
import cv2                # OpenCV
import pylab as pl        # pylab pour l'affichage
import numpy as np        # numpy pour les calculs sur matrices
from scipy import ndimage # scipy pour les convolution
from pylab import plot


#%%
# Chargement des données

data_dir = '/home/viki/data_td/perception/TD1/matching'
images_name = data_dir+'/features2d_%d.jpg'
patterns_name = data_dir+'/features2d_%s.jpg'
patterns_type = ['java', 'opengl', 'qt4']
images = [cv2.imread(images_name%i,cv2.IMREAD_GRAYSCALE) for i in range(1,8)]
patterns = [cv2.imread(patterns_name%i,cv2.IMREAD_GRAYSCALE) for i in patterns_type]

I = images[0]


#%%
###########################################
# Extraction des points caracteristiques  #
###########################################
#
# Travailler sur l'image 0 et 1 de la collection d'image
# et chercher à les mettre en correspondance
#
# L'objectif va etre d'extraire sur la paire d'image des points
# characteristiques, de calculer des descripteurs, et de mettre en correspondance
# ceux-ci
#
#
print('Extraction et matching de points')

#%%
#############################################################################
# Question : trouver les detecteur disponnible dans opencv
#            pour cela allez sur la page de documentation d'opencv 2.4 :
# http://docs.opencv.org/2.4/modules/features2d/doc/common_interfaces_of_feature_detectors.html
############################################################################
# Reponse
detecteurs_type = []

#
# Meme question pour les descripteurs :
# http://docs.opencv.org/2.4/modules/features2d/doc/common_interfaces_of_descriptor_extractors.html

descripteur_type = []

############################################################################

#%%
# classer les constructeurs
detecteurs = {}
descripteurs = {}
############################################################################
# Question : testez les detecteurs sur l'image I et afficher le nombre de
# features trouvées et construisez un dictionnaire de classe de detecteurs
# ou on accède par le nom du détecteur en majuscule.
############################################################################
# Reponse

for key in detecteurs_type:
    print('detecteur %s trouvé %d features'%(key,??)))
#%%
############################################################################
# Question : même question sur les descripteurs, en les calculant sur les
#            points characteristiques kp suivant :
kp = detecteurs['FAST'].detect(I)
############################################################################
# Reponse


for desc in descripteur_type:
    print('descripteur %s calculer %d features'%(desc,??))

############################################################################
#%%

def drawMatches(img1, kp1, img2, kp2, matches):
    """
    img1,img2 - Grayscale images
    kp1,kp2 - Detected list of keypoints through any of the OpenCV keypoint
              detection algorithms
    matches - A list of matches of corresponding keypoints through any
              OpenCV keypoint matching algorithm
    """
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]
    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')
    out[:rows1,:cols1] = np.dstack([img1, img1, img1])
    out[:rows2,cols1:] = np.dstack([img2, img2, img2])
    for mat in matches:
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt
        cv2.circle(out, (int(x1),int(y1)), 4, (255, 0, 0), 1)
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (255, 0, 0), 1)
        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (255, 0, 0), 1)
    return out
#%%
class Matcheur:
    # attribus statiques
    descripteurs = descripteurs
    detecteurs = detecteurs

    # attribus de la classe Matcheur
    pts0 = None
    pts1 = None
    I0 = None
    I1 = None
    inlier_mask = None
    seuil_ransac = 20

    # Methodes statiques de la classe
    @staticmethod
    def GetAviableDetector():
        return Matcheur.detecteurs
    @staticmethod
    def GetAviableDescriptor():
        return Matcheur.descripteurs
    @staticmethod
    def getPointsFromMatch(match, kp1, kp2):
        pts0 = np.array([kp1[m.queryIdx].pt for m in match])
        pts1 = np.array([kp2[m.trainIdx].pt for m in match])
        return pts0, pts1


    def __init__(self,detecteur = 'FAST', descripteur = 'BRIEF'):
        '''
             Constructeur de la classe, on y construit les classes membres que sont
             le détecteur, le descripteur ainsi que la classe d'affectation grâce aux
             vecteurs d'introspection précédement calculé
        '''
        if not self.detecteurs.has_key(detecteur.upper()):
            raise TypeError('detecteur inconnue :'+str(detecteur))
        if not self.descripteurs.has_key(descripteur.upper()):
            raise TypeError('descripteur  inconnue'+str(descripteur))
        self.detecteur_name = detecteur.upper()
        self.descripteur_name = descripteur.upper()
        self.detector = self.detecteurs[self.detecteur_name]
        self.descriptor = self.descripteurs[self.descripteur_name]
        self.matcheur = cv2.BFMatcher(cv2.NORM_L2, crossCheck = True )

    def match(self, I0, I1):
        '''
            Methode effectuant la mise en correspondance
        '''
        self.I0 = I0
        self.I1 = I1
        self.inlier_mask = None

        # Detecter les points dans I0 et I1
        self.kp0 = self.detector.detect(I0)
        self.kp1 = self.detector.detect(I1)

        # Calculer les descripteurs dans I0 et I1
        self.kp0, self.desc0 = self.descriptor.compute(I0,self.kp0)
        self.kp1, self.desc1 = self.descriptor.compute(I1,self.kp1)

        # Calcul de la mise en correspondance entre les descripteurs
        self.matches = self.matcheur.match(self.desc0, self.desc1)

        ###################################################################
        # Question : utiliser une lamda expression et la fonction sorted
        # pour trier les elements de self.matches en fonction de leurs
        # attribut 'distance'
        ####################################################################
        # Reponse :
        self.matches =
        ####################################################################

        self.pts0, self.pts1 = Matcheur.getPointsFromMatch(self.matches, self.kp0, self.kp1)
        return self.pts0, self.pts1

    def affine_match(self,motion_model = 'homography'):
        '''
            Fonction rafinant la mise en correspondance en utilisant un
            model géométrique à priorit, les models sont :
             'homography' : model projectif correspondant au déplacement d'un plan
             'fundamental' : model correspondant a la contrainte epipolaire entre
                             deux vues
            affine_match(self,motion_model = 'homography') -> flag, pts0, pts1, M
            flag = booleen si le model est valide
            pts0, pts1 = points inliers dans les images 0 et 1
            M = matrice du model geometrique estime
        '''
        if self.pts0 is None:
            raise RuntimeError('Veuillez appeler match(I0,I1) avant de rafiner')

        if motion_model is 'homography':
            findModel = cv2.findHomography
        elif motion_model is 'fundamental':
            findModel = cv2.findFundamentalMat
        else:
            raise TypeError('model '+str(motion_model)+' inconus')

        pts0, pts1 = Matcheur.getPointsFromMatch(self.matches, self.kp0, self.kp1)
        M, mask = findModel(pts0, pts1, cv2.RANSAC, 5.0)
        ####################################################################
        # Question : compter le nombre d'inlier / outlier du model
        ####################################################################
        # Reponse :
        inlier =
        outlier =
        ####################################################################
        print 'inliers  : ', inlier
        print 'outliers : ', outlier

        if inlier > self.seuil_ransac:
            is_good = True
            #####################################################################
            #  Question : selectionner les points inliers en utilisant la
            #             methode ravel de mask pour le serialiser
            ####################################################################
            # Reponse :
            pts0 =
            pts1 =
            ####################################################################
            self.inlier_mask = mask
            self.pts0 = pts0
            self.pts1 = pts1
            return is_good, self.pts0, self.pts1, M
        else:
            self.inlier_mask = mask
            return False, None, None, None

    def print_parameters(self):
        print 'detecteur        : ', self.detecteur_name
        print 'descripteur      : ', self.descripteur_name
        print 'seuil good match : ', self.seuil_match

    def show_current(self):
        '''
            Fonction affichant la mise en correspondance actuelle
        '''
        if self.pts0 is not None:
            print 'nombre de match : ', len(self.matches)
            if self.inlier_mask is not None :
                valid = [ m for idx, m in enumerate(self.matches) if self.inlier_mask[idx] == 1]
            else:
                valid = self.matches
            img = drawMatches(self.I0,self.kp0,self.I1,self.kp1,valid)
            figure()
            imshow(img)
        else:
            print 'pas de match!'

#%%
#
# Traitement de données...
#
# Pour afficher des temps de calcul dans le terminal ipython
# il suffit d'utiliser la commante :
# %timit LA_COMMANDE_PYTHON
#
# Dans la section qui va suivre l'objectif est de comprendre les compromis
# en temps de calcul et en robustesse des différents descripteurs
#
# Pour cela on va se concentrer sur des detecteurs 'fast'
#

###################################################################
# Question : créer deux instance de la classe matcheur pour le
#            parametres 'ORB'/'ORB'  et 'surf'/'surf'
###################################################################
# Reponse :
matcheur_orb   =
matcheur_surf  =
###################################################################

#%%
#
# Evaluer les temps de mise en correspondances pour les deux matcheurs
# sur deux images d'une meme séquence
#
I0 = images[0]
I1 = images[1]

####################################################################
# Question : evaluer les temps de calcul des deux commandes suivante
#           en utilisant la fonctionnalité %timeit de ipython
###################################################################

pts0, pts1 = matcheur_orb.match(I0,I1)
pts0, pts1 = matcheur_surf.match(I0,I1)
# Reponse :
#  temps de calcul orb :
#  temps de calcul surf :

#%%
#
# Evaluation du resultat
#

matcheur_orb.show_current()
title('ORB avant nettoyage')

#%%
is_valid, pts0, pts1, M = matcheur_orb.affine_match('homography')
matcheur_orb.show_current()
title('ORB apres nettoyage')

#%%

matcheur_surf.show_current()
title('SURF avant nettoyage')
is_valid, pts0, pts1, M = matcheur_surf.affine_match('homography')
matcheur_surf.show_current()
title('SURF apres nettoyage')

###########################################################################
#  Question : les mises en correponsance ont-elle réussit? la difference
#              de cout de calcul vous semble-t-elle justifié?
###########################################################################
# Reponse :
#
###########################################################################

#%%
#
# Réeffectuer le test en prennant une image et un patterns de livre
#

I0 = images[0]
I1 = patterns[1]

##########################################################################

pts0, pts1 = matcheur_orb.match(I0,I1)
matcheur_orb.show_current()
title('ORB avant nettoyage')
#%%
is_valid, pts0, pts1, M = matcheur_orb.affine_match('homography')
matcheur_orb.show_current()
title('ORB apres nettoyage')

#%%
pts0, pts1 = matcheur_surf.match(I0,I1)
matcheur_surf.show_current()
title('SURF avant nettoyage')
#%%
is_valid, pts0, pts1, M = matcheur_surf.affine_match('homography')
matcheur_surf.show_current()
title('SURF apres nettoyage')

###########################################################################
#  Question :  Que remarquez vous? quel est la difficulté introduit par
#              l'utilisation d'une image provenant de patterns?
#              les mises en correponsance ont-elle réussit? la difference
#              de cout de calcul vous semble-t-elle justifié?
###########################################################################
# Reponse :
#
###########################################################################









