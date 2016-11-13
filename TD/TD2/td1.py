# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 19:15:21 2015

@author: Aurélien Plyer
"""
# importation des modules qu'on va utiliser : 
import cv2                # OpenCV
import pylab as pl        # pylab pour l'affichage
import numpy as np        # numpy pour les calculs sur matrices
from scipy import ndimage # scipy pour les convolution
from pylab import figure, imshow

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
    # attribus de la classe Matcheur
    pts0 = None
    pts1 = None
    I0 = None
    I1 = None
    inlier_mask = None
    seuil_ransac = 20
    
    # Methodes statiques de la classe
    @staticmethod
    def InitStatic():
        detecteurs_type = ['FAST', 'STAR', 'SIFT', 'SURF', 'ORB', 'BRISK', 'MSER', 'GFTT', 
              'HARRIS', 'Dense', 'SimpleBlob']
        descripteur_type = ['SIFT','SURF', 'BRIEF', 'BRISK','ORB','FREAK']
        Matcheur.detecteurs = {}
        Matcheur.descripteurs = {}
        for key in detecteurs_type:
            detect = cv2.FeatureDetector_create(key)
            Matcheur.detecteurs[key.upper()] = detect
        for key in descripteur_type:
            descripteur = cv2.DescriptorExtractor_create(key)
            Matcheur.descripteurs[key.upper()] = descripteur
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
        
    
    def __init__(self,detecteur = 'ORB', descripteur = 'ORB'):
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
        self.matches = sorted(self.matches, key = lambda x: x.distance)
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
        def stereoModel(pts0, pts1, compat1, compat2):
            error = np.abs(pts1[:,1]-pts0[:,1])
            mask = np.ones([pts1.shape[0],1])
            mask[error > 1] = 0
            return None, mask
        if self.pts0 is None:
            raise RuntimeError('Veuillez appeler match(I0,I1) avant de rafiner')
        
        if motion_model is 'homography':
            findModel = cv2.findHomography
        elif motion_model is 'fundamental':
            findModel = cv2.findFundamentalMat
        elif motion_model is 'stereo':
            findModel = stereoModel
        else:
            raise TypeError('model '+str(motion_model)+' inconus')
        
        pts0, pts1 = Matcheur.getPointsFromMatch(self.matches, self.kp0, self.kp1)
        M, mask = findModel(pts0, pts1, cv2.RANSAC, 5.0)
        ####################################################################
        # Question : compter le nombre d'inlier / outlier du model
        ####################################################################
        # Reponse : 
        inlier = np.sum(mask == 1)
        outlier = np.sum(mask == 0)
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
            pts0 = pts0[mask.ravel() == 1]
            pts1 = pts1[mask.ravel() == 1]
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
    def get_valid_kp(self):
        if self.inlier_mask is not None :
            valid = [ m for idx, m in enumerate(self.matches) if self.inlier_mask[idx] == 1]
        else:
            valid = self.matches
        kp0 = [self.kp0[m.queryIdx] for m in valid]
        kp1 = [self.kp1[m.trainIdx] for m in valid]
        pts0 = np.array([self.kp0[m.queryIdx].pt for m in valid])
        pts1 = np.array([self.kp1[m.trainIdx].pt for m in valid])
        desc0 = np.array([self.desc0[m.queryIdx] for m in valid])
        desc1 = np.array([self.desc1[m.trainIdx] for m in valid])
        return pts0, pts1, kp0, kp1, desc0, desc1
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

